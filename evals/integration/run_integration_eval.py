#!/usr/bin/env python3
"""Integration evaluation runner for scicode-lint.

Runs multi-pattern integration tests on realistic code scenarios
and validates overall detection capabilities.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scicode_lint import SciCodeLinter


class IntegrationEvaluator:
    """Evaluator for multi-pattern integration tests."""

    def __init__(self, config_path: Path):
        """Initialize evaluator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.scenarios_dir = config_path.parent / "scenarios"
        self.linter = SciCodeLinter()

    def _load_config(self) -> dict[str, Any]:
        """Load expected findings configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)  # type: ignore[no-any-return]

    def run_all_scenarios(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios (async wrapper)."""
        return asyncio.run(self._run_all_scenarios_async(verbose))

    async def _run_all_scenarios_async(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios concurrently."""
        overall: dict[str, Any] = {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "total_bugs_injected": 0,
            "total_bugs_found": 0,
            "false_positives": 0,
        }
        results: dict[str, Any] = {
            "scenarios": {},
            "overall": overall,
        }

        # Build list of (scenario_name, scenario_config) pairs
        scenarios = list(self.config["scenarios"].items())

        if verbose:
            print(f"\nRunning {len(scenarios)} scenarios concurrently...")

        # Run all scenarios in parallel
        tasks = [self._run_scenario_async(name, config, verbose) for name, config in scenarios]
        scenario_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for (scenario_name, _), raw_result in zip(scenarios, scenario_results):
            if isinstance(raw_result, Exception):
                result: dict[str, Any] = {
                    "passed": False,
                    "error": str(raw_result),
                    "expected_count": 0,
                    "found_count": 0,
                    "false_positives": 0,
                }
            else:
                result = cast(dict[str, Any], raw_result)

            results["scenarios"][scenario_name] = result

            # Update overall stats
            overall["total_scenarios"] += 1
            if result["passed"]:
                overall["passed"] += 1
            else:
                overall["failed"] += 1

            overall["total_bugs_injected"] += result["expected_count"]
            overall["total_bugs_found"] += result["found_count"]
            overall["false_positives"] += result["false_positives"]

        # Calculate overall metrics
        # true_positives = expected bugs that were actually found (not extra findings)
        total_true_positives = sum(
            r.get("true_positives", 0) for r in results["scenarios"].values()
        )
        total_injected: int = overall["total_bugs_injected"]
        total_found: int = overall["total_bugs_found"]
        extra_findings = total_found - total_true_positives - overall["false_positives"]

        overall["true_positives"] = total_true_positives
        overall["extra_findings"] = extra_findings

        if total_injected > 0:
            # Recall = expected bugs found / expected bugs (capped at 100%)
            overall["recall"] = total_true_positives / total_injected
        else:
            overall["recall"] = 0.0

        if total_found > 0:
            # Precision = true positives / total findings
            overall["precision"] = total_true_positives / total_found
        else:
            overall["precision"] = 0.0

        return results

    async def _run_scenario_async(
        self, scenario_name: str, scenario_config: dict[str, Any], verbose: bool = False
    ) -> dict[str, Any]:
        """Run a single integration test scenario asynchronously."""
        file_path = self.scenarios_dir / scenario_config["file"].split("/")[-1]

        if not file_path.exists():
            return {
                "passed": False,
                "error": f"Scenario file not found: {file_path}",
                "expected_count": 0,
                "found_count": 0,
                "false_positives": 0,
            }

        if verbose:
            print(f"[{scenario_name}] File: {file_path}")

        # Run linter async - directly use the async method
        try:
            result = await self.linter._check_file_async(file_path)
            findings = result.findings
        except Exception as e:
            return {
                "passed": False,
                "error": f"Linter error: {str(e)}",
                "expected_count": 0,
                "found_count": 0,
                "false_positives": 0,
            }

        # Process findings (same as sync version)
        return self._process_scenario_results(scenario_name, scenario_config, findings, verbose)

    def _process_scenario_results(
        self,
        scenario_name: str,
        scenario_config: dict[str, Any],
        findings: list[Any],
        verbose: bool,
    ) -> dict[str, Any]:
        """Process scenario results and return metrics."""
        # Count findings by pattern
        findings_by_pattern: defaultdict[str, int] = defaultdict(int)
        for finding in findings:
            findings_by_pattern[finding.id] += 1

        # Compare against expected
        expected_patterns = scenario_config.get("expected_patterns", {})
        expected_total = sum(expected_patterns.values())
        found_total = len(findings)

        # Check each expected pattern
        pattern_matches = {}
        missing_patterns = []
        extra_patterns = []
        true_positives = 0  # Expected bugs that were found

        for pattern_id, expected_count in expected_patterns.items():
            found_count = findings_by_pattern.get(pattern_id, 0)
            # True positives for this pattern = min(found, expected)
            tp_for_pattern = min(found_count, expected_count)
            true_positives += tp_for_pattern

            pattern_matches[pattern_id] = {
                "expected": expected_count,
                "found": found_count,
                "match": found_count == expected_count,
            }

            if found_count < expected_count:
                missing_patterns.append(
                    f"{pattern_id} (expected {expected_count}, found {found_count})"
                )
            elif found_count > expected_count:
                extra_patterns.append(
                    f"{pattern_id} (expected {expected_count}, found {found_count})"
                )

        # Check for unexpected patterns (false positives)
        false_positive_patterns = []
        for pattern_id, count in findings_by_pattern.items():
            if pattern_id not in expected_patterns:
                false_positive_patterns.append(f"{pattern_id} ({count} findings)")

        false_positives = sum(
            count
            for pattern_id, count in findings_by_pattern.items()
            if pattern_id not in expected_patterns
        )

        # Determine if scenario passed: all expected found, zero false positives
        passed = false_positives == 0 and len(missing_patterns) == 0

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{scenario_name}] Exp: {expected_total}, Found: {found_total}, {status}")
            if false_positive_patterns:
                print(f"[{scenario_name}] False positives: {', '.join(false_positive_patterns)}")

        return {
            "passed": passed,
            "expected_count": expected_total,
            "found_count": found_total,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "pattern_matches": pattern_matches,
            "missing_patterns": missing_patterns,
            "extra_patterns": extra_patterns,
            "false_positive_patterns": false_positive_patterns,
            "findings": [
                {
                    "id": f.id,
                    "lines": f.location.lines if f.location else [],
                    "snippet": f.location.snippet if f.location else "",
                    "confidence": f.confidence,
                }
                for f in findings
            ],
        }

    def run_scenario(
        self, scenario_name: str, scenario_config: dict[str, Any], verbose: bool = False
    ) -> dict[str, Any]:
        """Run a single integration test scenario (sync wrapper)."""
        file_path = self.scenarios_dir / scenario_config["file"].split("/")[-1]

        if not file_path.exists():
            return {
                "passed": False,
                "error": f"Scenario file not found: {file_path}",
                "expected_count": 0,
                "found_count": 0,
                "false_positives": 0,
            }

        if verbose:
            print(f"File: {file_path}")
            print(f"Description: {scenario_config['description']}")

        # Run linter
        try:
            result = self.linter.check_file(file_path)
            findings = result.findings
        except Exception as e:
            return {
                "passed": False,
                "error": f"Linter error: {str(e)}",
                "expected_count": 0,
                "found_count": 0,
                "false_positives": 0,
            }

        return self._process_scenario_results(scenario_name, scenario_config, findings, verbose)

    def generate_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate markdown report."""
        lines = ["# Integration Test Results\n"]

        # Overall summary
        overall = results["overall"]
        lines.append("## Overall Summary\n")
        lines.append(f"- **Scenarios**: {overall['passed']}/{overall['total_scenarios']} passed")
        true_positives = overall["true_positives"]
        bugs_injected = overall["total_bugs_injected"]
        lines.append(
            f"- **Recall**: {overall['recall']:.1%} "
            f"({true_positives}/{bugs_injected} expected bugs detected)"
        )
        lines.append(f"- **Precision**: {overall['precision']:.1%}")
        lines.append(f"- **False Positives**: {overall['false_positives']}")
        extra = overall.get("extra_findings", 0)
        if extra > 0:
            lines.append(f"- **Extra Findings**: {extra} (findings beyond expected count)")
        lines.append("")

        # Check thresholds: require 100% recall and zero false positives
        lines.append("## Threshold Checks\n")
        recall_pass = overall["recall"] == 1.0
        fp_pass = overall["false_positives"] == 0

        recall_status = "✓ PASS" if recall_pass else "✗ FAIL"
        fp_status = "✓ PASS" if fp_pass else "✗ FAIL"
        lines.append(f"- Recall = 100%: {recall_status} ({overall['recall']:.1%})")
        lines.append(f"- False positives = 0: {fp_status} ({overall['false_positives']})\n")

        # Per-scenario results
        lines.append("## Scenario Results\n")

        for scenario_name, scenario_result in results["scenarios"].items():
            status = "✓ PASS" if scenario_result["passed"] else "✗ FAIL"
            lines.append(f"### {scenario_name} - {status}\n")

            if "error" in scenario_result:
                lines.append(f"**Error**: {scenario_result['error']}\n")
                continue

            lines.append(f"- Expected: {scenario_result['expected_count']} bugs")
            lines.append(f"- Found: {scenario_result['found_count']} bugs")
            lines.append(f"- False positives: {scenario_result['false_positives']}\n")

            # Pattern breakdown
            if scenario_result.get("pattern_matches"):
                lines.append("**Pattern breakdown:**\n")
                for pattern_id, match in scenario_result["pattern_matches"].items():
                    status_mark = "✓" if match["match"] else "✗"
                    lines.append(
                        f"- {status_mark} `{pattern_id}`: {match['found']}/{match['expected']}"
                    )
                lines.append("")

            if scenario_result.get("missing_patterns"):
                lines.append("**Missing patterns:**\n")
                for pattern in scenario_result["missing_patterns"]:
                    lines.append(f"- {pattern}")
                lines.append("")

            if scenario_result.get("false_positive_patterns"):
                lines.append("**False positives:**\n")
                for pattern in scenario_result["false_positive_patterns"]:
                    lines.append(f"- {pattern}")
                lines.append("")

        # Write report
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"\nReport written to: {output_path}")


def main() -> int:
    """Run integration evaluations."""
    parser = argparse.ArgumentParser(description="Run integration evaluations")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run specific scenario only",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "expected_findings.yaml",
        help="Path to expected findings config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "integration_report.md",
        help="Output report path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Also output JSON results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    evaluator = IntegrationEvaluator(args.config)

    # Run evaluations
    if args.scenario:
        # Run single scenario
        scenario_config = evaluator.config["scenarios"].get(args.scenario)
        if not scenario_config:
            print(f"Error: Scenario '{args.scenario}' not found")
            sys.exit(1)

        result = evaluator.run_scenario(args.scenario, scenario_config, args.verbose)

        # Calculate metrics for single scenario
        total_injected = result["expected_count"]
        total_found = result["found_count"]
        true_positives = result["true_positives"]
        false_positives = result["false_positives"]

        if total_injected > 0:
            recall = true_positives / total_injected
        else:
            recall = 0.0

        if total_found > 0:
            precision = true_positives / total_found
        else:
            precision = 0.0

        results = {
            "scenarios": {args.scenario: result},
            "overall": {
                "total_scenarios": 1,
                "passed": 1 if result["passed"] else 0,
                "failed": 0 if result["passed"] else 1,
                "total_bugs_injected": total_injected,
                "total_bugs_found": total_found,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "extra_findings": total_found - true_positives - false_positives,
                "recall": recall,
                "precision": precision,
            },
        }
    else:
        # Run all scenarios
        results = evaluator.run_all_scenarios(args.verbose)

    # Generate reports
    evaluator.generate_report(results, args.output)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON results written to: {args.json}")

    # Exit with appropriate code
    if results["overall"]["failed"] > 0:
        sys.exit(1)
    else:
        print("\n✓ All integration tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
