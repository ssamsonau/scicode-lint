"""
Dynamic Integration Evaluation

Generates fresh test files with intentional bugs, runs linter,
and uses LLM-as-judge to evaluate results.

Categories:
- TP-intended: Encoded bug was detected
- TP-bonus: Real bug found that wasn't intentionally encoded
- FP: Non-existent bug was reported
- FN: Encoded bug was missed
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from scicode_lint import SciCodeLinter
from scicode_lint.detectors.catalog import DetectionCatalog


def _build_pattern_list() -> str:
    """Build pattern list from registry for the generator prompt."""
    # Get patterns directory
    patterns_dir = Path(__file__).parent.parent.parent / "patterns"
    catalog = DetectionCatalog(patterns_dir)

    # Group patterns by category
    by_category: dict[str, list[str]] = {}
    for pattern in catalog.patterns:
        cat = pattern.category
        if cat not in by_category:
            by_category[cat] = []
        # Format: - pattern-id: brief description from warning_message
        brief = pattern.warning_message.split(".")[0] if pattern.warning_message else pattern.id
        by_category[cat].append(f"- {pattern.id}: {brief}")

    # Build formatted string
    lines = []
    for category in sorted(by_category.keys()):
        lines.append(f"\n{category}:")
        lines.extend(sorted(by_category[category]))

    return "\n".join(lines)


@dataclass
class IntentionalBug:
    """A bug intentionally inserted into generated code."""

    bug_id: str
    pattern_id: str
    line_number: int
    description: str


@dataclass
class GeneratedScenario:
    """A generated test scenario with code and bug manifest."""

    code: str
    bugs: list[IntentionalBug]
    scenario_description: str


@dataclass
class LinterDetection:
    """A detection reported by the linter."""

    pattern_id: str
    line_number: int | None
    message: str
    confidence: float


@dataclass
class JudgmentResult:
    """Result of LLM judge evaluation."""

    tp_intended: list[dict[str, str]] = field(default_factory=list)
    tp_bonus: list[dict[str, str]] = field(default_factory=list)
    fp: list[dict[str, str]] = field(default_factory=list)
    fn: list[dict[str, str]] = field(default_factory=list)
    judge_reasoning: str = ""


GENERATOR_PROMPT_TEMPLATE = """You are a test case generator for scicode-lint, an AI-powered linter \
for scientific Python code.

Generate a realistic Python file that contains EXACTLY the specified bugs. The code should:
1. Look like real ML/scientific research code (not contrived)
2. Be 50-150 lines
3. Contain the intentional bugs naturally embedded in realistic code
4. NOT contain any comments or hints about the bugs

Available patterns to insert bugs for:
{pattern_list}

TASK: Generate code with {num_bugs} intentional bugs from the patterns above.

Respond in this exact JSON format:
```json
{{
  "scenario_description": "Brief description of what this code does",
  "bugs": [
    {{
      "bug_id": "bug_1",
      "pattern_id": "the-pattern-id",
      "line_number": 42,
      "description": "What the bug is"
    }}
  ],
  "code": "the full Python code here"
}}
```

Make the code realistic - a training pipeline, data preprocessing script, or inference function."""

JUDGE_PROMPT = """You are evaluating a linter's detection results.

## Generated Code
```python
{code}
```

## Intentional Bugs (Ground Truth)
{manifest}

## Linter Detections
{detections}

## Your Task

Evaluate each linter detection and each intentional bug. Categorize into:

1. **TP-intended**: An intentional bug that was correctly detected
   - Match by pattern_id AND approximate line location (within 5 lines)

2. **TP-bonus**: A REAL bug the linter found that wasn't in the manifest
   - Must be an actual code quality issue, not a style preference
   - Be strict: only count if it's genuinely problematic

3. **FP (False Positive)**: A detection that is NOT a real bug
   - The code is actually correct, or the issue doesn't apply

4. **FN (False Negative)**: An intentional bug that was NOT detected
   - Check if any detection matches the bug (pattern + location)

Respond in this exact JSON format:
```json
{{
  "tp_intended": [
    {{"bug_id": "bug_1", "pattern_id": "...", "detection_line": 42, "reasoning": "..."}}
  ],
  "tp_bonus": [
    {{"pattern_id": "...", "line": 50, "reasoning": "Why this is a real bug"}}
  ],
  "fp": [
    {{"pattern_id": "...", "line": 30, "reasoning": "Why this is not a bug"}}
  ],
  "fn": [
    {{"bug_id": "bug_2", "pattern_id": "...", "line": 60, "reasoning": "Why it was missed"}}
  ],
  "summary": "Overall assessment"
}}
```

Be rigorous. When in doubt about TP-bonus, classify as FP (conservative)."""


class DynamicEvaluator:
    """Orchestrates dynamic evaluation with LLM generation and judging."""

    def __init__(
        self,
        linter_command: list[str] | None = None,
    ):
        self.linter_command = linter_command or [
            "python",
            "-m",
            "scicode_lint",
            "check",
        ]
        self.linter = SciCodeLinter()  # Direct library usage for async

    async def _call_llm_async(self, prompt: str) -> str:
        """Call Claude via CLI asynchronously."""
        cmd = ["claude", "--print", "-p", prompt]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {stderr.decode()}")
        return stdout.decode()

    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call Claude via CLI (sync fallback)."""
        cmd = ["claude", "--print", "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout

    async def generate_scenario_async(self, num_bugs: int = 3) -> GeneratedScenario:
        """Generate a test scenario with intentional bugs asynchronously."""
        pattern_list = _build_pattern_list()
        prompt = GENERATOR_PROMPT_TEMPLATE.format(num_bugs=num_bugs, pattern_list=pattern_list)
        response = await self._call_llm_async(prompt)

        # Extract JSON from response
        json_str = self._extract_json(response)
        data = json.loads(json_str)

        bugs = [
            IntentionalBug(
                bug_id=b["bug_id"],
                pattern_id=b["pattern_id"],
                line_number=b["line_number"],
                description=b["description"],
            )
            for b in data["bugs"]
        ]

        return GeneratedScenario(
            code=data["code"],
            bugs=bugs,
            scenario_description=data["scenario_description"],
        )

    def generate_scenario(self, num_bugs: int = 3) -> GeneratedScenario:
        """Generate a test scenario with intentional bugs."""
        pattern_list = _build_pattern_list()
        prompt = GENERATOR_PROMPT_TEMPLATE.format(num_bugs=num_bugs, pattern_list=pattern_list)
        response = self._call_llm(prompt)

        # Extract JSON from response
        json_str = self._extract_json(response)
        data = json.loads(json_str)

        bugs = [
            IntentionalBug(
                bug_id=b["bug_id"],
                pattern_id=b["pattern_id"],
                line_number=b["line_number"],
                description=b["description"],
            )
            for b in data["bugs"]
        ]

        return GeneratedScenario(
            code=data["code"],
            bugs=bugs,
            scenario_description=data["scenario_description"],
        )

    async def run_linter_async(self, code: str) -> list[LinterDetection]:
        """Run scicode-lint on the generated code asynchronously."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            # Use library directly for async execution
            result = await self.linter._check_file_async(temp_path)

            detections = []
            for finding in result.findings:
                loc = finding.location
                line_num = loc.lines[0] if loc and loc.lines else 0
                detections.append(
                    LinterDetection(
                        pattern_id=finding.id,
                        line_number=line_num,
                        message=finding.explanation or "",
                        confidence=finding.confidence,
                    )
                )
            return detections
        finally:
            temp_path.unlink(missing_ok=True)

    def run_linter(self, code: str) -> list[LinterDetection]:
        """Run scicode-lint on the generated code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            cmd = self.linter_command + [temp_path, "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0 and not result.stdout:
                print(f"Linter error: {result.stderr}")
                return []

            detections = []
            try:
                output = json.loads(result.stdout)
                # Output is a list of file results, each with findings
                if isinstance(output, list):
                    for file_result in output:
                        for finding in file_result.get("findings", []):
                            detections.append(
                                LinterDetection(
                                    pattern_id=finding.get("id", ""),
                                    line_number=finding.get("location", {}).get("lines", [0])[0],
                                    message=finding.get("explanation", ""),
                                    confidence=finding.get("confidence", 0.0),
                                )
                            )
                elif isinstance(output, dict):
                    for finding in output.get("findings", []):
                        detections.append(
                            LinterDetection(
                                pattern_id=finding.get("id", ""),
                                line_number=finding.get("location", {}).get("lines", [0])[0],
                                message=finding.get("explanation", ""),
                                confidence=finding.get("confidence", 0.0),
                            )
                        )
            except json.JSONDecodeError:
                print(f"Failed to parse linter output: {result.stdout[:200]}")

            return detections
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def judge_results_async(
        self,
        scenario: GeneratedScenario,
        detections: list[LinterDetection],
    ) -> JudgmentResult:
        """Use LLM to judge the linter's results asynchronously."""
        manifest = json.dumps(
            [
                {
                    "bug_id": b.bug_id,
                    "pattern_id": b.pattern_id,
                    "line": b.line_number,
                    "description": b.description,
                }
                for b in scenario.bugs
            ],
            indent=2,
        )

        detections_str = json.dumps(
            [
                {
                    "pattern_id": d.pattern_id,
                    "line": d.line_number,
                    "message": d.message,
                    "confidence": d.confidence,
                }
                for d in detections
            ],
            indent=2,
        )

        prompt = JUDGE_PROMPT.format(
            code=scenario.code,
            manifest=manifest,
            detections=detections_str,
        )

        response = await self._call_llm_async(prompt)
        json_str = self._extract_json(response)
        data = json.loads(json_str)

        return JudgmentResult(
            tp_intended=data.get("tp_intended", []),
            tp_bonus=data.get("tp_bonus", []),
            fp=data.get("fp", []),
            fn=data.get("fn", []),
            judge_reasoning=data.get("summary", ""),
        )

    def judge_results(
        self,
        scenario: GeneratedScenario,
        detections: list[LinterDetection],
    ) -> JudgmentResult:
        """Use LLM to judge the linter's results."""
        manifest = json.dumps(
            [
                {
                    "bug_id": b.bug_id,
                    "pattern_id": b.pattern_id,
                    "line": b.line_number,
                    "description": b.description,
                }
                for b in scenario.bugs
            ],
            indent=2,
        )

        detections_str = json.dumps(
            [
                {
                    "pattern_id": d.pattern_id,
                    "line": d.line_number,
                    "message": d.message,
                    "confidence": d.confidence,
                }
                for d in detections
            ],
            indent=2,
        )

        prompt = JUDGE_PROMPT.format(
            code=scenario.code,
            manifest=manifest,
            detections=detections_str,
        )

        response = self._call_llm(prompt)
        json_str = self._extract_json(response)
        data = json.loads(json_str)

        return JudgmentResult(
            tp_intended=data.get("tp_intended", []),
            tp_bonus=data.get("tp_bonus", []),
            fp=data.get("fp", []),
            fn=data.get("fn", []),
            judge_reasoning=data.get("summary", ""),
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from markdown code blocks or raw text."""
        import re

        # Try to find JSON in code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find raw JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError(f"No JSON found in response: {text[:200]}...")

    async def _run_single_scenario_async(
        self, scenario_num: int, num_scenarios: int, bugs_per_scenario: int
    ) -> dict[str, Any]:
        """Run a single scenario asynchronously."""
        print(f"[Scenario {scenario_num}/{num_scenarios}] Generating...")

        try:
            scenario = await self.generate_scenario_async(bugs_per_scenario)
            print(f"[Scenario {scenario_num}] {scenario.scenario_description}")

            detections = await self.run_linter_async(scenario.code)
            print(f"[Scenario {scenario_num}] Detections: {len(detections)}")

            judgment = await self.judge_results_async(scenario, detections)

            result = {
                "description": scenario.scenario_description,
                "bugs_inserted": len(scenario.bugs),
                "detections": len(detections),
                "tp_intended": len(judgment.tp_intended),
                "tp_bonus": len(judgment.tp_bonus),
                "fp": len(judgment.fp),
                "fn": len(judgment.fn),
                "reasoning": judgment.judge_reasoning,
            }
            print(
                f"[Scenario {scenario_num}] TP={result['tp_intended']}, "
                f"Bonus={result['tp_bonus']}, FP={result['fp']}, FN={result['fn']}"
            )
            return result
        except Exception as e:
            print(f"[Scenario {scenario_num}] Error: {e}")
            return {
                "description": f"Error: {e}",
                "bugs_inserted": 0,
                "detections": 0,
                "tp_intended": 0,
                "tp_bonus": 0,
                "fp": 0,
                "fn": 0,
                "reasoning": str(e),
            }

    def run_evaluation(self, num_scenarios: int = 10, bugs_per_scenario: int = 3) -> dict[str, Any]:
        """Run full dynamic evaluation (async wrapper)."""
        return asyncio.run(self._run_evaluation_async(num_scenarios, bugs_per_scenario))

    async def _run_evaluation_async(
        self, num_scenarios: int = 10, bugs_per_scenario: int = 3
    ) -> dict[str, Any]:
        """Run full dynamic evaluation asynchronously."""
        print(f"\nRunning {num_scenarios} scenarios concurrently...")

        # Run all scenarios in parallel
        tasks = [
            self._run_single_scenario_async(i + 1, num_scenarios, bugs_per_scenario)
            for i in range(num_scenarios)
        ]
        scenario_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        results: dict[str, Any] = {
            "scenarios": [],
            "totals": {
                "tp_intended": 0,
                "tp_bonus": 0,
                "fp": 0,
                "fn": 0,
            },
        }

        for raw_result in scenario_results:
            if isinstance(raw_result, Exception):
                result: dict[str, Any] = {
                    "description": f"Error: {raw_result}",
                    "bugs_inserted": 0,
                    "detections": 0,
                    "tp_intended": 0,
                    "tp_bonus": 0,
                    "fp": 0,
                    "fn": 0,
                    "reasoning": str(raw_result),
                }
            else:
                result = cast(dict[str, Any], raw_result)

            results["scenarios"].append(result)
            results["totals"]["tp_intended"] += result["tp_intended"]
            results["totals"]["tp_bonus"] += result["tp_bonus"]
            results["totals"]["fp"] += result["fp"]
            results["totals"]["fn"] += result["fn"]

        # Calculate metrics
        totals = results["totals"]
        tp = totals["tp_intended"] + totals["tp_bonus"]
        fp = totals["fp"]
        fn = totals["fn"]

        results["metrics"] = {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall_intended": (
                totals["tp_intended"] / (totals["tp_intended"] + fn)
                if (totals["tp_intended"] + fn) > 0
                else 0.0
            ),
            "bonus_rate": (totals["tp_bonus"] / num_scenarios),
        }

        return results


def main() -> None:
    """Run dynamic evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic integration evaluation")
    parser.add_argument("--scenarios", type=int, default=10, help="Number of scenarios to generate")
    parser.add_argument("--bugs", type=int, default=3, help="Bugs per scenario")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    evaluator = DynamicEvaluator()
    results = evaluator.run_evaluation(
        num_scenarios=args.scenarios,
        bugs_per_scenario=args.bugs,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Scenarios run: {args.scenarios}")
    print(f"Total TP (intended): {results['totals']['tp_intended']}")
    print(f"Total TP (bonus): {results['totals']['tp_bonus']}")
    print(f"Total FP: {results['totals']['fp']}")
    print(f"Total FN: {results['totals']['fn']}")
    print(f"\nPrecision: {results['metrics']['precision']:.1%}")
    print(f"Recall (intended bugs): {results['metrics']['recall_intended']:.1%}")
    print(f"Bonus finds per scenario: {results['metrics']['bonus_rate']:.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
