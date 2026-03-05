#!/usr/bin/env python3
"""Integration evaluation runner for scicode-lint.

Runs multi-pattern integration tests on realistic code scenarios
and validates overall detection capabilities.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
            return yaml.safe_load(f)

    def run_all_scenarios(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios."""
        results = {
            "scenarios": {},
            "overall": {
                "total_scenarios": 0,
                "passed": 0,
                "failed": 0,
                "total_bugs_injected": 0,
                "total_bugs_found": 0,
                "false_positives": 0,
            },
        }

        for scenario_name, scenario_config in self.config["scenarios"].items():
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Running scenario: {scenario_name}")
                print(f"{'=' * 60}")

            result = self.run_scenario(scenario_name, scenario_config, verbose)
            results["scenarios"][scenario_name] = result

            # Update overall stats
            results["overall"]["total_scenarios"] += 1
            if result["passed"]:
                results["overall"]["passed"] += 1
            else:
                results["overall"]["failed"] += 1

            results["overall"]["total_bugs_injected"] += result["expected_count"]
            results["overall"]["total_bugs_found"] += result["found_count"]
            results["overall"]["false_positives"] += result["false_positives"]

        # Calculate overall metrics
        total_injected = results["overall"]["total_bugs_injected"]
        total_found = results["overall"]["total_bugs_found"]

        if total_injected > 0:
            results["overall"]["coverage"] = total_found / total_injected
            results["overall"]["recall"] = total_found / total_injected
        else:
            results["overall"]["coverage"] = 0.0
            results["overall"]["recall"] = 0.0

        if total_found > 0:
            correct = total_found - results["overall"]["false_positives"]
            results["overall"]["precision"] = correct / total_found
        else:
            results["overall"]["precision"] = 0.0

        return results

    def run_scenario(
        self, scenario_name: str, scenario_config: dict[str, Any], verbose: bool = False
    ) -> dict[str, Any]:
        """Run a single integration test scenario."""
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

        # Count findings by pattern
        findings_by_pattern = defaultdict(int)
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

        for pattern_id, expected_count in expected_patterns.items():
            found_count = findings_by_pattern.get(pattern_id, 0)
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

        # Determine if scenario passed
        min_total = scenario_config.get("min_total_findings", expected_total)
        max_total = scenario_config.get("max_total_findings", expected_total)
        max_fp = scenario_config.get("max_false_positives", 0)

        passed = (
            found_total >= min_total
            and found_total <= max_total
            and false_positives <= max_fp
            and len(missing_patterns) == 0
        )

        if verbose:
            print(f"\nExpected: {expected_total} findings")
            print(f"Found: {found_total} findings")
            print(f"Status: {'PASS' if passed else 'FAIL'}")

            if pattern_matches:
                print("\nPattern breakdown:")
                for pattern_id, match in pattern_matches.items():
                    status = "✓" if match["match"] else "✗"
                    print(f"  {status} {pattern_id}: {match['found']}/{match['expected']}")

            if missing_patterns:
                print(f"\nMissing patterns: {', '.join(missing_patterns)}")

            if false_positive_patterns:
                print(f"\nFalse positives: {', '.join(false_positive_patterns)}")

        return {
            "passed": passed,
            "expected_count": expected_total,
            "found_count": found_total,
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

    def generate_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate markdown report."""
        lines = ["# Integration Test Results\n"]

        # Overall summary
        overall = results["overall"]
        lines.append("## Overall Summary\n")
        lines.append(f"- **Scenarios**: {overall['passed']}/{overall['total_scenarios']} passed")
        lines.append(
            f"- **Coverage**: {overall['coverage']:.1%} ({overall['total_bugs_found']}/{overall['total_bugs_injected']} bugs detected)"
        )
        lines.append(f"- **Precision**: {overall['precision']:.1%}")
        lines.append(f"- **Recall**: {overall['recall']:.1%}")
        lines.append(f"- **False Positives**: {overall['false_positives']}\n")

        # Check thresholds
        thresholds = self.config.get("thresholds", {})
        min_coverage = thresholds.get("min_coverage", 0.9)

        lines.append("## Threshold Checks\n")
        coverage_pass = overall["coverage"] >= min_coverage
        fp_pass = overall["false_positives"] == 0

        lines.append(
            f"- Coverage ≥ {min_coverage:.0%}: {'✓ PASS' if coverage_pass else '✗ FAIL'} ({overall['coverage']:.1%})"
        )
        lines.append(
            f"- False positives = 0: {'✓ PASS' if fp_pass else '✗ FAIL'} ({overall['false_positives']})\n"
        )

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


def main():
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
        false_positives = result["false_positives"]

        if total_injected > 0:
            coverage = total_found / total_injected
            recall = total_found / total_injected
        else:
            coverage = 0.0
            recall = 0.0

        if total_found > 0:
            correct = total_found - false_positives
            precision = correct / total_found
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
                "false_positives": false_positives,
                "coverage": coverage,
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
