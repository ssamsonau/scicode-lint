#!/usr/bin/env python3
"""Integration evaluation with LLM-as-judge for semantic correctness.

Similar to pattern evals, this uses an LLM judge to evaluate whether
findings semantically match expected bugs, allowing for more flexible
evaluation beyond exact pattern ID matching.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scicode_lint import SciCodeLinter
from scicode_lint.config import LLMConfig
from scicode_lint.llm.client import create_client

JUDGE_SYSTEM_PROMPT = """You are evaluating the correctness of a code linter's bug detection.

Your task:
1. Compare the expected bug description against the linter's actual finding
2. Determine if the linter correctly identified this specific bug
3. Output verdict as JSON: yes, no, or partial

Guidelines:
- "yes" = Linter correctly identified this bug with accurate location/reasoning
- "no" = Linter missed this bug or found a different issue
- "partial" = Linter identified something related but incomplete/unclear

Focus on semantic correctness. The linter should detect the same underlying
bug even if using different words or pattern names.

CRITICAL: Output ONLY valid JSON, nothing else - NO markdown fences, NO extra text.
Format: {"verdict": "yes"|"no"|"partial", "reasoning": "brief explanation", "confidence": 0.95}
"""


def generate_judge_prompt(
    expected_bug: dict[str, Any],
    finding: dict[str, Any],
    code_context: str,
) -> str:
    """Generate prompt for LLM judge to evaluate a finding."""
    return f"""<EXPECTED_BUG>
Line: {expected_bug.get("line", "unknown")}
Pattern: {expected_bug["pattern"]}
Description: {expected_bug["description"]}
</EXPECTED_BUG>

<LINTER_FINDING>
Pattern ID: {finding["id"]}
Location: {finding["location"]}
Confidence: {finding["confidence"]}
Explanation: {finding["explanation"][:300]}...
</LINTER_FINDING>

<CODE_CONTEXT>
{code_context}
</CODE_CONTEXT>

Does the linter finding correctly identify the expected bug?
Consider:
- Is it the same underlying issue?
- Is the location approximately correct?
- Is the reasoning sound?

Output ONLY valid JSON:
{{"verdict": "yes"|"no"|"partial", "reasoning": "brief explanation", "confidence": 0.95}}
"""


class IntegrationJudgeEvaluator:
    """Evaluator for integration tests using LLM-as-judge."""

    def __init__(self, config_path: Path):
        """Initialize evaluator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.scenarios_dir = config_path.parent / "scenarios"
        self.linter = SciCodeLinter()

        # Create LLM client for judge
        llm_config = LLMConfig()
        self.judge_llm = create_client(llm_config)

    def _load_config(self) -> dict[str, Any]:
        """Load expected findings configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)  # type: ignore[no-any-return]

    async def evaluate_finding(
        self,
        expected_bug: dict[str, Any],
        finding: dict[str, Any],
        code_context: str,
    ) -> dict[str, Any]:
        """Use LLM judge to evaluate if finding matches expected bug."""
        prompt = generate_judge_prompt(expected_bug, finding, code_context)

        try:
            response = await self.judge_llm.generate(  # type: ignore[attr-defined]
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=200,
            )

            # Parse JSON response
            result = json.loads(response.strip())
            return {
                "verdict": result.get("verdict", "no"),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.0),
            }

        except Exception as e:
            print(f"Warning: Judge evaluation failed: {e}")
            return {
                "verdict": "no",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.0,
            }

    def run_scenario(
        self,
        scenario_name: str,
        scenario_config: dict[str, Any],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a single integration test scenario with LLM judge."""
        file_path = self.scenarios_dir / scenario_config["file"].split("/")[-1]

        if not file_path.exists():
            return {
                "passed": False,
                "error": f"Scenario file not found: {file_path}",
                "evaluated_bugs": [],
            }

        if verbose:
            print(f"\nEvaluating: {scenario_name}")
            print(f"File: {file_path}")
            print(f"Description: {scenario_config['description']}")

        # Read code context
        code_context = file_path.read_text()

        # Run linter
        try:
            result = self.linter.check_file(file_path)
            findings = result.findings
        except Exception as e:
            return {
                "passed": False,
                "error": f"Linter error: {str(e)}",
                "evaluated_bugs": [],
            }

        # Get expected bugs
        expected_bugs = scenario_config.get("bugs", [])

        if verbose:
            print(f"\nExpected bugs: {len(expected_bugs)}")
            print(f"Linter findings: {len(findings)}")

        # Evaluate each expected bug
        evaluated_bugs = []

        # TODO: This should be async but keeping sync for now
        # In a real implementation, use asyncio.gather to evaluate all in parallel
        import asyncio

        async def evaluate_all() -> None:
            for expected_bug in expected_bugs:
                # Find best matching finding for this bug
                best_match = None
                best_score = 0.0

                # Simple heuristic: match by pattern ID first
                expected_pattern = expected_bug["pattern"]
                for finding in findings:
                    loc = finding.location
                    loc_name = getattr(loc, "name", str(loc)) if loc else "unknown"
                    finding_dict = {
                        "id": finding.id,
                        "location": loc_name,
                        "confidence": finding.confidence,
                        "explanation": finding.explanation,
                    }

                    # Prefer same pattern ID
                    if finding.id == expected_pattern:
                        best_match = finding_dict
                        best_score = 0.9
                        break
                    elif best_score < 0.5:
                        best_match = finding_dict
                        best_score = 0.5

                if best_match:
                    evaluation = await self.evaluate_finding(
                        expected_bug, best_match, code_context[:1000]
                    )
                    evaluated_bugs.append(
                        {
                            "expected": expected_bug,
                            "finding": best_match,
                            "judge_verdict": evaluation["verdict"],
                            "judge_reasoning": evaluation["reasoning"],
                        }
                    )
                else:
                    # No finding for this expected bug
                    evaluated_bugs.append(
                        {
                            "expected": expected_bug,
                            "finding": None,
                            "judge_verdict": "no",
                            "judge_reasoning": "No matching finding",
                        }
                    )

        # Run async evaluation
        asyncio.run(evaluate_all())

        # Calculate metrics
        correct = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "yes")
        partial = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "partial")
        incorrect = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "no")

        recall = correct / len(expected_bugs) if expected_bugs else 0.0
        partial_credit_recall = (
            (correct + 0.5 * partial) / len(expected_bugs) if expected_bugs else 0.0
        )

        passed = recall >= 0.8  # 80% recall threshold

        if verbose:
            print("\nJudge Results:")
            print(f"  Correct: {correct}/{len(expected_bugs)}")
            print(f"  Partial: {partial}/{len(expected_bugs)}")
            print(f"  Missed: {incorrect}/{len(expected_bugs)}")
            print(f"  Recall: {recall:.1%}")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")

        return {
            "passed": passed,
            "expected_count": len(expected_bugs),
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "recall": recall,
            "partial_credit_recall": partial_credit_recall,
            "evaluated_bugs": evaluated_bugs,
        }

    def run_all_scenarios(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios with LLM judge."""
        overall: dict[str, Any] = {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "total_bugs": 0,
            "correct": 0,
            "partial": 0,
            "incorrect": 0,
        }
        results: dict[str, Any] = {
            "scenarios": {},
            "overall": overall,
        }

        for scenario_name, scenario_config in self.config["scenarios"].items():
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Scenario: {scenario_name}")
                print(f"{'=' * 60}")

            result = self.run_scenario(scenario_name, scenario_config, verbose)
            results["scenarios"][scenario_name] = result

            # Update overall stats
            overall["total_scenarios"] += 1
            if result["passed"]:
                overall["passed"] += 1
            else:
                overall["failed"] += 1

            overall["total_bugs"] += result.get("expected_count", 0)
            overall["correct"] += result.get("correct", 0)
            overall["partial"] += result.get("partial", 0)
            overall["incorrect"] += result.get("incorrect", 0)

        # Calculate overall metrics
        total: int = overall["total_bugs"]
        if total > 0:
            overall["recall"] = overall["correct"] / total
            overall["partial_credit_recall"] = (
                overall["correct"] + 0.5 * overall["partial"]
            ) / total
        else:
            overall["recall"] = 0.0
            overall["partial_credit_recall"] = 0.0

        return results

    def generate_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate markdown report."""
        lines = ["# Integration Test Results (LLM-as-Judge)\n"]

        # Overall summary
        overall = results["overall"]
        lines.append("## Overall Summary\n")
        lines.append(f"- **Scenarios**: {overall['passed']}/{overall['total_scenarios']} passed")
        correct = overall['correct']
        total_bugs = overall['total_bugs']
        lines.append(
            f"- **Recall**: {overall['recall']:.1%} "
            f"({correct}/{total_bugs} bugs detected)"
        )
        lines.append(f"- **Partial Credit Recall**: {overall['partial_credit_recall']:.1%}")
        lines.append(f"- **Correct**: {overall['correct']} bugs")
        lines.append(f"- **Partial**: {overall['partial']} bugs")
        lines.append(f"- **Missed**: {overall['incorrect']} bugs\n")

        # Per-scenario results
        lines.append("## Scenario Results\n")

        for scenario_name, scenario_result in results["scenarios"].items():
            status = "✓ PASS" if scenario_result["passed"] else "✗ FAIL"
            lines.append(f"### {scenario_name} - {status}\n")

            if "error" in scenario_result:
                lines.append(f"**Error**: {scenario_result['error']}\n")
                continue

            lines.append(f"- Recall: {scenario_result['recall']:.1%}")
            lines.append(
                f"- Correct: {scenario_result['correct']}/{scenario_result['expected_count']}"
            )
            lines.append(
                f"- Partial: {scenario_result['partial']}/{scenario_result['expected_count']}"
            )
            lines.append(
                f"- Missed: {scenario_result['incorrect']}/{scenario_result['expected_count']}\n"
            )

        # Write report
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"\nReport written to: {output_path}")


def main() -> int:
    """Run integration evaluations with LLM judge."""
    parser = argparse.ArgumentParser(description="Run integration evaluations with LLM-as-judge")
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
        default=Path(__file__).parent / "integration_judge_report.md",
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

    evaluator = IntegrationJudgeEvaluator(args.config)

    # Run evaluations
    if args.scenario:
        # Run single scenario
        scenario_config = evaluator.config["scenarios"].get(args.scenario)
        if not scenario_config:
            print(f"Error: Scenario '{args.scenario}' not found")
            sys.exit(1)

        result = evaluator.run_scenario(args.scenario, scenario_config, args.verbose)
        results = {
            "scenarios": {args.scenario: result},
            "overall": {
                "total_scenarios": 1,
                "passed": 1 if result["passed"] else 0,
                "failed": 0 if result["passed"] else 1,
                "total_bugs": result.get("expected_count", 0),
                "correct": result.get("correct", 0),
                "partial": result.get("partial", 0),
                "incorrect": result.get("incorrect", 0),
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
