#!/usr/bin/env python3
"""Integration evaluation with LLM-as-judge for semantic correctness.

Similar to pattern evals, this uses an LLM judge to evaluate whether
findings semantically match expected bugs, allowing for more flexible
evaluation beyond exact pattern ID matching.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from scicode_lint import SciCodeLinter
from scicode_lint.config import LLMConfig
from scicode_lint.llm.client import create_client


class MatchVerdict(BaseModel):
    """Judge verdict on whether a finding matches an expected bug."""

    verdict: str = Field(description="yes, no, or partial")
    reasoning: str = Field(description="Brief explanation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")


class ExtraFindingVerdict(BaseModel):
    """Judge verdict on whether an extra finding is a real bug or false positive."""

    verdict: str = Field(description="real_bug or false_positive")
    reasoning: str = Field(description="Brief explanation")


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

EXTRA_FINDING_JUDGE_PROMPT = """You are evaluating whether a linter finding is a REAL bug or a FALSE POSITIVE.

The linter found something that wasn't in our expected bug list. Your job is to determine
if this is actually a legitimate code quality issue (TP-bonus) or a false alarm (FP).

CRITICAL: Be strict. Only mark as "real_bug" if the code genuinely has this problem.
Common real issues in ML code:
- Hardcoded hyperparameters without config (lr=0.001, epochs=50)
- Missing random seeds for reproducibility
- Data leakage between train/test
- Missing model.train()/model.eval() calls
- Missing optimizer.zero_grad()

CRITICAL: Output ONLY valid JSON, nothing else - NO markdown fences, NO extra text.
Format: {"verdict": "real_bug"|"false_positive", "reasoning": "brief explanation"}
"""


def generate_extra_finding_prompt(finding: dict[str, Any], code_context: str) -> str:
    """Generate prompt for evaluating an extra (unexpected) finding."""
    return f"""<LINTER_FINDING>
Pattern ID: {finding["id"]}
Location: {finding["location"]}
Confidence: {finding["confidence"]}
Explanation: {finding["explanation"]}
</LINTER_FINDING>

<CODE_CONTEXT>
{code_context}
</CODE_CONTEXT>

Is this finding a REAL bug in the code, or a FALSE POSITIVE?

Look at the actual code and determine:
1. Does the code actually have this issue?
2. Is this a legitimate code quality problem?
3. Would fixing this improve the code?

Output ONLY valid JSON:
{{"verdict": "real_bug"|"false_positive", "reasoning": "brief explanation"}}
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

        # Create LLM client for judge (uses vLLM like pattern eval)
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
            verdict = await self.judge_llm.async_complete_structured(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=MatchVerdict,
            )
            return {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
                "confidence": verdict.confidence,
            }

        except Exception as e:
            print(f"Warning: Judge evaluation failed: {e}")
            return {
                "verdict": "no",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.0,
            }

    async def evaluate_extra_finding(
        self,
        finding: dict[str, Any],
        code_context: str,
    ) -> dict[str, Any]:
        """Use LLM judge to evaluate if an extra finding is a real bug or FP."""
        prompt = generate_extra_finding_prompt(finding, code_context)

        try:
            verdict = await self.judge_llm.async_complete_structured(
                system_prompt=EXTRA_FINDING_JUDGE_PROMPT,
                user_prompt=prompt,
                schema=ExtraFindingVerdict,
            )
            return {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
            }

        except Exception as e:
            print(f"Warning: Extra finding evaluation failed: {e}")
            # Default to false_positive to be conservative
            return {
                "verdict": "false_positive",
                "reasoning": f"Error: {str(e)}",
            }

    def run_scenario(
        self,
        scenario_name: str,
        scenario_config: dict[str, Any],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a single integration test scenario with LLM judge."""
        return asyncio.run(self._run_scenario_async(scenario_name, scenario_config, verbose))

    async def _run_scenario_async(
        self,
        scenario_name: str,
        scenario_config: dict[str, Any],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a single integration test scenario with LLM judge (async)."""
        file_path = self.scenarios_dir / scenario_config["file"].split("/")[-1]

        if not file_path.exists():
            return {
                "passed": False,
                "error": f"Scenario file not found: {file_path}",
                "evaluated_bugs": [],
                "extra_findings": [],
            }

        if verbose:
            print(f"\n[{scenario_name}] Evaluating: {file_path.name}")

        # Read code context
        code_context = file_path.read_text()

        # Run linter
        try:
            result = await self.linter._check_file_async(file_path)
            findings = result.findings
        except Exception as e:
            return {
                "passed": False,
                "error": f"Linter error: {str(e)}",
                "evaluated_bugs": [],
                "extra_findings": [],
            }

        # Get expected bugs
        expected_bugs = scenario_config.get("bugs", [])

        if verbose:
            print(f"[{scenario_name}] Expected: {len(expected_bugs)}, Found: {len(findings)}")

        # Convert findings to dicts and track which are matched
        finding_dicts = []
        matched_finding_ids = set()

        for i, finding in enumerate(findings):
            loc = finding.location
            loc_name = getattr(loc, "name", str(loc)) if loc else "unknown"
            finding_dicts.append(
                {
                    "index": i,
                    "id": finding.id,
                    "location": loc_name,
                    "confidence": finding.confidence,
                    "explanation": finding.explanation or "",
                }
            )

        # Evaluate each expected bug
        evaluated_bugs = []

        for expected_bug in expected_bugs:
            expected_pattern = expected_bug["pattern"]

            # Find best matching finding
            best_match = None
            best_match_idx = None

            for fd in finding_dicts:
                if fd["index"] in matched_finding_ids:
                    continue  # Already matched to another bug

                if fd["id"] == expected_pattern:
                    best_match = fd
                    best_match_idx = fd["index"]
                    break

            if best_match:
                matched_finding_ids.add(best_match_idx)
                evaluation = await self.evaluate_finding(
                    expected_bug, best_match, code_context[:2000]
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
                evaluated_bugs.append(
                    {
                        "expected": expected_bug,
                        "finding": None,
                        "judge_verdict": "no",
                        "judge_reasoning": "No matching finding",
                    }
                )

        # Evaluate EXTRA findings (not matched to expected bugs)
        extra_findings = []
        unmatched = [fd for fd in finding_dicts if fd["index"] not in matched_finding_ids]

        if unmatched and verbose:
            print(f"[{scenario_name}] Evaluating {len(unmatched)} extra findings...")

        # Evaluate extra findings in parallel
        if unmatched:
            extra_tasks = [self.evaluate_extra_finding(fd, code_context[:2000]) for fd in unmatched]
            extra_results = await asyncio.gather(*extra_tasks)

            for fd, extra_result in zip(unmatched, extra_results):
                if isinstance(extra_result, Exception):
                    verdict_info = {"verdict": "false_positive", "reasoning": str(extra_result)}
                else:
                    verdict_info = extra_result
                extra_findings.append(
                    {
                        "finding": fd,
                        "judge_verdict": verdict_info["verdict"],
                        "judge_reasoning": verdict_info["reasoning"],
                    }
                )

        # Calculate metrics
        correct = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "yes")
        partial = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "partial")
        incorrect = sum(1 for eb in evaluated_bugs if eb["judge_verdict"] == "no")

        tp_bonus = sum(1 for ef in extra_findings if ef["judge_verdict"] == "real_bug")
        fp = sum(1 for ef in extra_findings if ef["judge_verdict"] == "false_positive")

        recall = correct / len(expected_bugs) if expected_bugs else 0.0
        partial_credit_recall = (
            (correct + 0.5 * partial) / len(expected_bugs) if expected_bugs else 0.0
        )

        # Precision: (correct + bonus) / (correct + bonus + fp)
        total_positives = correct + tp_bonus + fp
        precision = (correct + tp_bonus) / total_positives if total_positives > 0 else 1.0

        # Pass requires all expected bugs found and zero false positives
        passed = recall == 1.0 and fp == 0

        if verbose:
            print(f"[{scenario_name}] Recall: {recall:.0%}, TP-bonus: {tp_bonus}, FP: {fp}")

        return {
            "passed": passed,
            "expected_count": len(expected_bugs),
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "tp_bonus": tp_bonus,
            "false_positives": fp,
            "recall": recall,
            "partial_credit_recall": partial_credit_recall,
            "precision": precision,
            "evaluated_bugs": evaluated_bugs,
            "extra_findings": extra_findings,
        }

    def run_all_scenarios(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios with LLM judge (async wrapper)."""
        return asyncio.run(self._run_all_scenarios_async(verbose))

    async def _run_all_scenarios_async(self, verbose: bool = False) -> dict[str, Any]:
        """Run all integration test scenarios with LLM judge concurrently."""
        overall: dict[str, Any] = {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "total_bugs": 0,
            "correct": 0,
            "partial": 0,
            "incorrect": 0,
            "tp_bonus": 0,
            "false_positives": 0,
        }
        results: dict[str, Any] = {
            "scenarios": {},
            "overall": overall,
        }

        scenarios = list(self.config["scenarios"].items())

        if verbose:
            print(f"\nRunning {len(scenarios)} scenarios concurrently with LLM judge...")

        # Run all scenarios in parallel
        tasks = [self._run_scenario_async(name, config, verbose) for name, config in scenarios]
        scenario_results = await asyncio.gather(*tasks, return_exceptions=True)

        for (scenario_name, _), raw_result in zip(scenarios, scenario_results):
            if isinstance(raw_result, Exception):
                result: dict[str, Any] = {
                    "passed": False,
                    "error": str(raw_result),
                    "expected_count": 0,
                    "correct": 0,
                    "partial": 0,
                    "incorrect": 0,
                    "tp_bonus": 0,
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

            overall["total_bugs"] += result.get("expected_count", 0)
            overall["correct"] += result.get("correct", 0)
            overall["partial"] += result.get("partial", 0)
            overall["incorrect"] += result.get("incorrect", 0)
            overall["tp_bonus"] += result.get("tp_bonus", 0)
            overall["false_positives"] += result.get("false_positives", 0)

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

        # Calculate precision: (correct + bonus) / (correct + bonus + fp)
        total_positives = overall["correct"] + overall["tp_bonus"] + overall["false_positives"]
        if total_positives > 0:
            overall["precision"] = (overall["correct"] + overall["tp_bonus"]) / total_positives
        else:
            overall["precision"] = 1.0

        return results

    def generate_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate markdown report."""
        lines = ["# Integration Test Results (LLM-as-Judge)\n"]

        # Overall summary
        overall = results["overall"]
        lines.append("## Overall Summary\n")
        lines.append(f"- **Scenarios**: {overall['passed']}/{overall['total_scenarios']} passed")

        correct = overall["correct"]
        total_bugs = overall["total_bugs"]
        lines.append(
            f"- **Recall**: {overall.get('recall', 0):.1%} ({correct}/{total_bugs} expected bugs)"
        )
        lines.append(f"- **Precision**: {overall.get('precision', 0):.1%}")
        lines.append(f"- **TP-bonus** (real bugs not in expected): {overall.get('tp_bonus', 0)}")
        lines.append(f"- **False Positives**: {overall.get('false_positives', 0)}\n")

        lines.append("## Metrics Breakdown\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Expected bugs detected | {correct}/{total_bugs} |")
        lines.append(f"| Partial detections | {overall['partial']} |")
        lines.append(f"| Missed bugs | {overall['incorrect']} |")
        lines.append(f"| Bonus true positives | {overall.get('tp_bonus', 0)} |")
        lines.append(f"| False positives | {overall.get('false_positives', 0)} |\n")

        # Per-scenario results
        lines.append("## Scenario Results\n")

        for scenario_name, scenario_result in results["scenarios"].items():
            status = "✓ PASS" if scenario_result.get("passed") else "✗ FAIL"
            lines.append(f"### {scenario_name} - {status}\n")

            if "error" in scenario_result:
                lines.append(f"**Error**: {scenario_result['error']}\n")
                continue

            exp_count = scenario_result.get("expected_count", 0)
            lines.append(f"- Recall: {scenario_result.get('recall', 0):.1%}")
            lines.append(f"- Correct: {scenario_result.get('correct', 0)}/{exp_count}")
            lines.append(f"- Partial: {scenario_result.get('partial', 0)}/{exp_count}")
            lines.append(f"- Missed: {scenario_result.get('incorrect', 0)}/{exp_count}")
            lines.append(f"- TP-bonus: {scenario_result.get('tp_bonus', 0)}")
            lines.append(f"- False positives: {scenario_result.get('false_positives', 0)}\n")

            # Show extra findings detail
            extra = scenario_result.get("extra_findings", [])
            if extra:
                real_bugs = [e for e in extra if e["judge_verdict"] == "real_bug"]
                fps = [e for e in extra if e["judge_verdict"] == "false_positive"]

                if real_bugs:
                    lines.append("**Bonus detections (real bugs):**\n")
                    for e in real_bugs:
                        lines.append(f"- `{e['finding']['id']}`: {e['judge_reasoning']}")
                    lines.append("")

                if fps:
                    lines.append("**False positives:**\n")
                    for e in fps:
                        lines.append(f"- `{e['finding']['id']}`: {e['judge_reasoning']}")
                    lines.append("")

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
