#!/usr/bin/env python3
"""Integration evaluation for scicode-lint.

Full pipeline (no human-in-the-loop):
1. GENERATE (Sonnet) - Select patterns, generate code with bugs
2. VERIFY (Sonnet) - Confirm manifest is accurate
3. LINT (vLLM) - Run scicode-lint on generated code
4. JUDGE (Sonnet) - Categorize findings as TP/FP/FN

Modes:
- Ephemeral (default): Generate, evaluate, print report, discard
- Persistent (--save): Generate, evaluate, save to generated/<run-id>/
- Re-evaluate (--id exists): Load saved scenarios, re-run lint + judge

Usage:
    # Full pipeline - generate and evaluate
    python integration_eval.py --generate-count 10

    # Save with custom ID
    python integration_eval.py --generate-count 10 --save --id baseline_v1

    # Re-evaluate existing ID (no generation, just lint + judge)
    python integration_eval.py --id baseline_v1

    # List saved runs
    python integration_eval.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from dev_lib.claude_cli import ClaudeCLI, ClaudeCLIError
from dev_lib.run_output import RunOutput, write_worker
from scicode_lint import SciCodeLinter
from scicode_lint.detectors.catalog import DetectionCatalog

# Directories
INTEGRATION_DIR = Path(__file__).parent
GENERATED_DIR = INTEGRATION_DIR / "generated"
REPORTS_DIR = INTEGRATION_DIR / "reports"


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class PatternSelection(BaseModel):
    """Result of pattern selection by LLM."""

    patterns: list[str] = Field(description="List of selected pattern IDs (e.g., pt-001, ml-002)")
    scenario_type: str = Field(description="Brief description of the scenario type")
    reasoning: str = Field(description="Why these patterns work together")


class ManifestEntry(BaseModel):
    """Single bug entry in the manifest."""

    pattern_id: str = Field(description="Pattern ID (e.g., pt-001)")
    line: int = Field(description="Line number where the bug occurs")
    description: str = Field(description="Brief description of the bug instance")


class GeneratedScenario(BaseModel):
    """Result of scenario generation by LLM."""

    code: str = Field(description="Complete Python code containing the bugs")
    manifest: list[ManifestEntry] = Field(description="List of bugs in the code")


class VerificationEntry(BaseModel):
    """Verification result for a single bug."""

    pattern_id: str = Field(description="Pattern ID being verified")
    line: int = Field(description="Claimed line number")
    correct: bool = Field(description="Whether the bug is actually present")
    actual_line: int | None = Field(default=None, description="Corrected line number if different")
    notes: str = Field(default="", description="Notes about the verification")


class VerificationResult(BaseModel):
    """Result of manifest verification."""

    verified: list[VerificationEntry] = Field(description="Verification for each claimed bug")
    quality: str = Field(description="good, needs_correction, or regenerate")
    corrected_manifest: list[ManifestEntry] | None = Field(
        default=None, description="Corrected manifest if needed"
    )


class JudgedFinding(BaseModel):
    """Judge evaluation of a single linter finding."""

    pattern_id: str = Field(description="Pattern ID from the finding")
    line: int = Field(description="Line number from the finding")
    category: str = Field(description="tp_intended, tp_bonus, or fp")
    reasoning: str = Field(description="Why this categorization")


class JudgedMiss(BaseModel):
    """Judge evaluation of a missed bug."""

    pattern_id: str = Field(description="Pattern ID that was missed")
    line: int = Field(description="Line number where bug should be")
    reasoning: str = Field(description="Why it was missed or if it's actually present")


class JudgeResult(BaseModel):
    """Result of LLM judge evaluation."""

    findings: list[JudgedFinding] = Field(description="Evaluation of each linter finding")
    misses: list[JudgedMiss] = Field(description="Evaluation of missed bugs")
    # Note: summary accepts Any because judge sometimes returns dict instead of string
    summary: Any = Field(description="Overall assessment")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PatternInfo:
    """Pattern information from the catalog."""

    id: str
    description: str
    category: str


@dataclass
class ScenarioResult:
    """A generated scenario with evaluation results."""

    name: str
    code: str
    patterns: list[str]
    manifest: list[dict[str, Any]]
    verified: bool
    # Evaluation results
    bugs_intended: int = 0
    bugs_detected: int = 0
    tp_intended: int = 0
    tp_bonus: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    false_negatives: list[dict[str, Any]] = field(default_factory=list)


def categorize_judge_results(
    judge_result: JudgeResult | None,
    manifest: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    skip_judge: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Categorize linter findings into TP/FP/FN.

    Args:
        judge_result: LLM judge result (None if skip_judge)
        manifest: Expected bugs from scenario
        findings: Linter findings
        skip_judge: If True, use deterministic comparison

    Returns:
        Tuple of (tp_intended, tp_bonus, false_positives, false_negatives, log_lines)
    """
    log_lines: list[str] = []

    if skip_judge:
        expected_ids = {m["pattern_id"] for m in manifest}
        found_ids = {f["pattern_id"] for f in findings}

        tp_intended = [f for f in findings if f["pattern_id"] in expected_ids]
        tp_bonus = [f for f in findings if f["pattern_id"] not in expected_ids]
        false_positives: list[dict[str, Any]] = []
        false_negatives = [m for m in manifest if m["pattern_id"] not in found_ids]
        log_lines.append("[JUDGE] skipped (deterministic comparison)\n")
    else:
        assert judge_result is not None
        log_lines.append(f"[JUDGE]\n{judge_result.model_dump_json(indent=2)}\n")

        tp_intended = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "tp_intended"
        ]
        tp_bonus = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "tp_bonus"
        ]
        false_positives = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "fp"
        ]
        false_negatives = [
            {"pattern_id": m.pattern_id, "line": m.line, "reasoning": m.reasoning}
            for m in judge_result.misses
        ]

    return tp_intended, tp_bonus, false_positives, false_negatives, log_lines


# =============================================================================
# Prompts
# =============================================================================

SELECTION_SYSTEM = """You are an expert at selecting compatible bug patterns for integration testing.

Your task is to select 2-3 patterns that could naturally coexist in ONE realistic Python file.

COMPATIBILITY CRITERIA:
1. Patterns should make sense together (e.g., PyTorch patterns in a training script)
2. Avoid conflicting patterns (e.g., "missing model.eval()" AND "has model.eval()")
3. Prefer patterns from related categories when possible
4. Ensure the combined scenario is realistic scientific/ML code

Keep selection small (2-3 patterns) for focused, concise scenarios.
Output valid JSON only."""

SELECTION_USER = """Select 2-3 compatible patterns from this registry:

{registry}

Requirements:
- Pick 2-3 patterns that make sense in ONE Python file
- Consider what type of code would contain these bugs
- Prefer patterns that create a realistic scenario
- Keep it focused - fewer patterns = clearer test

Output JSON with: patterns (list of IDs), scenario_type (brief description), reasoning"""

GENERATION_SYSTEM = """You are an expert Python developer creating integration test scenarios.

Your task is to write realistic scientific/ML Python code that contains SPECIFIC bugs.

CRITICAL RULES:
1. Code must be realistic and runnable (with appropriate imports)
2. Code MUST contain the EXACT bugs listed - do NOT fix them
3. NO comments about bugs in the code - natural-looking code only
4. NO docstrings mentioning bugs, issues, or problems
5. Code should be 30-60 lines - keep it CONCISE
6. Use realistic variable names and structure
7. Keep output JSON compact - no extra whitespace

Output valid JSON only. Keep code SHORT to avoid truncation."""

GENERATION_USER = """Write a realistic {scenario_type} Python script.

The code MUST contain these EXACT bugs (do NOT fix them):

{pattern_descriptions}

Requirements:
1. Realistic, runnable Python code
2. Each bug must be clearly present at a specific line
3. NO comments or docstrings revealing the bugs
4. Natural variable names (not "buggy_function" or "leaky_scaler")

Output JSON with:
- code: The complete Python file content
- manifest: List of {{pattern_id, line, description}} for each bug"""

VERIFICATION_SYSTEM = """You are a code reviewer verifying bug manifests.

Your task is to check if code ACTUALLY contains the claimed bugs at the specified lines.

For each claimed bug:
1. Go to the specified line
2. Check if the bug pattern is actually present
3. If the line is wrong but bug exists elsewhere, note the correct line
4. If the bug doesn't exist at all, mark as incorrect

Output valid JSON only."""

VERIFICATION_USER = """Review this code and verify the manifest.

CODE:
```python
{code}
```

CLAIMED BUGS:
{manifest}

For each bug, verify:
1. Is the bug actually present in the code?
2. Is it at the claimed line number?
3. If line is wrong, what's the correct line?

Output JSON with:
- verified: List of {{pattern_id, line, correct, actual_line, notes}}
- quality: "good" (all correct), "needs_correction" (bugs exist but lines wrong), or "regenerate" (bugs missing)
- corrected_manifest: Corrected manifest if quality is "needs_correction\""""

JUDGE_SYSTEM = """You are evaluating a linter's detection results against a ground truth manifest.

Your task is to categorize each linter finding and identify missed bugs.

CATEGORIES:
1. tp_intended: Finding matches a bug in the manifest (same pattern, within 5 lines)
2. tp_bonus: Finding is a REAL bug not in the manifest - genuinely problematic code
3. fp: Finding is NOT a real bug - false positive, code is actually correct

Be strict about tp_bonus: only count if the code is genuinely problematic.
Be fair about fp: if the linter found a real issue, it's not a false positive.

Output valid JSON only."""

JUDGE_USER = """Evaluate these linter results.

## Code
```python
{code}
```

## Ground Truth Manifest (bugs that SHOULD be detected)
{manifest}

## Linter Findings (what the linter reported)
{findings}

For each linter finding, categorize as:
- tp_intended: Matches a manifest bug (pattern + approximate line)
- tp_bonus: Real bug NOT in manifest (bonus find)
- fp: Not a real bug (false positive)

For each manifest bug not matched by a finding, explain why it was missed.

Output JSON with:
- findings: List of {{pattern_id, line, category, reasoning}}
- misses: List of {{pattern_id, line, reasoning}} for unmatched manifest bugs
- summary: Overall assessment"""


# =============================================================================
# Generator Class
# =============================================================================


class ScenarioGenerator:
    """Generates and evaluates integration test scenarios using Claude CLI (Sonnet)."""

    def __init__(
        self,
        patterns_dir: Path | None = None,
        seed: int = 42,
    ):
        self.catalog = DetectionCatalog(patterns_dir)
        self.rng = random.Random(seed)
        self.patterns = self._load_pattern_registry()
        self.linter = SciCodeLinter()
        self.cli = ClaudeCLI(model="sonnet", effort="medium")

    def _load_pattern_registry(self) -> list[PatternInfo]:
        """Load all patterns with ID and description."""
        return [
            PatternInfo(
                id=p.id,
                description=p.warning_message,
                category=p.category,
            )
            for p in self.catalog.patterns
        ]

    def _format_registry(self, patterns: list[PatternInfo] | None = None) -> str:
        """Format pattern registry for LLM prompt."""
        patterns = patterns or self.patterns
        # Shuffle to avoid always selecting same patterns
        shuffled = self.rng.sample(patterns, len(patterns))
        lines = []
        for p in shuffled:
            lines.append(f"- {p.id} [{p.category}]: {p.description}")
        return "\n".join(lines)

    def _format_pattern_descriptions(self, pattern_ids: list[str]) -> str:
        """Format selected patterns for generation prompt."""
        lines = []
        for pid in pattern_ids:
            for p in self.patterns:
                if p.id == pid:
                    lines.append(f"- {p.id}: {p.description}")
                    break
        return "\n".join(lines)

    async def _call_claude(
        self, system_prompt: str, user_prompt: str, effort: str = "medium"
    ) -> dict[str, Any]:
        """Call Claude CLI and parse JSON response.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.
            effort: Thinking effort level.

        Returns:
            Parsed JSON dict from Claude response.

        Raises:
            ClaudeCLIError: On any Claude CLI error.
            ClaudeCLIParseError: If JSON parsing fails.
        """
        prompt = f"{system_prompt}\n\n{user_prompt}"
        return await self.cli.arun_json(prompt, effort=effort, timeout=180)

    async def select_patterns(self) -> PatternSelection:
        """Use Sonnet to select compatible patterns."""
        registry = self._format_registry()
        user_prompt = SELECTION_USER.format(registry=registry)

        result = await self._call_claude(SELECTION_SYSTEM, user_prompt)
        return PatternSelection.model_validate(result)

    async def generate_scenario(self, selection: PatternSelection) -> GeneratedScenario:
        """Use Sonnet to generate code with specified bugs."""
        pattern_descriptions = self._format_pattern_descriptions(selection.patterns)
        user_prompt = GENERATION_USER.format(
            scenario_type=selection.scenario_type,
            pattern_descriptions=pattern_descriptions,
        )

        result = await self._call_claude(GENERATION_SYSTEM, user_prompt)
        return GeneratedScenario.model_validate(result)

    async def verify_manifest(self, code: str, manifest: list[ManifestEntry]) -> VerificationResult:
        """Use Sonnet via Claude CLI to verify the manifest."""
        manifest_str = json.dumps([m.model_dump() for m in manifest], indent=2)
        prompt = (
            f"{VERIFICATION_SYSTEM}\n\n{VERIFICATION_USER.format(code=code, manifest=manifest_str)}"
        )

        try:
            content = await self.cli.arun_json(prompt, effort="high", timeout=120)
            return VerificationResult.model_validate(content)
        except ClaudeCLIError as e:
            print(f"Claude CLI error during verification: {e}")
            return VerificationResult(
                verified=[],
                quality="regenerate",
                corrected_manifest=None,
            )

    async def judge_findings(
        self, code: str, manifest: list[dict[str, Any]], findings: list[dict[str, Any]]
    ) -> JudgeResult:
        """Use Sonnet to judge linter findings against manifest.

        Args:
            code: The Python code that was analyzed
            manifest: Ground truth bugs (pattern_id, line, description)
            findings: Linter findings (pattern_id, line, message)

        Returns:
            JudgeResult with categorized findings and missed bugs
        """
        manifest_str = json.dumps(manifest, indent=2)
        findings_str = json.dumps(findings, indent=2)
        prompt = f"{JUDGE_SYSTEM}\n\n{JUDGE_USER.format(code=code, manifest=manifest_str, findings=findings_str)}"

        try:
            content = await self.cli.arun_json(prompt, effort="high", timeout=120)
            return JudgeResult.model_validate(content)
        except ClaudeCLIError as e:
            print(f"Claude CLI error during judging: {e}")
            return JudgeResult(findings=[], misses=[], summary=f"Claude error: {e}")

    async def run_linter(self, code: str) -> list[dict[str, Any]]:
        """Run linter on code and return findings as dicts.

        Args:
            code: The Python code to evaluate

        Returns:
            List of findings as dicts with pattern_id, line, message
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            temp_file = Path(f.name)

        try:
            result = await self.linter._check_file_async(temp_file)
            return [
                {
                    "pattern_id": f.id,
                    "line": f.location.lines[0] if f.location and f.location.lines else 0,
                    "message": f.explanation or "",
                }
                for f in result.findings
            ]
        finally:
            temp_file.unlink(missing_ok=True)

    async def generate_one(
        self,
        skip_verification: bool = False,
        skip_judge: bool = False,
        max_attempts: int = 3,
    ) -> tuple[ScenarioResult | None, str]:
        """Generate, verify, lint, and judge one scenario.

        Flow:
        1. Generate (Sonnet) - select patterns and generate code
        2. Verify manifest (Sonnet) - confirm bugs are present
        3. Run linter (vLLM) - detect bugs
        4. Judge results (Sonnet) - categorize findings

        Returns:
            Tuple of (ScenarioResult or None, raw_log) where raw_log contains
            serialized outputs from each step for disk persistence.
        """
        log_lines: list[str] = []

        for attempt in range(max_attempts):
            try:
                log_lines.append(f"=== Attempt {attempt + 1}/{max_attempts} ===\n")

                # Step 1: Generate (Sonnet)
                selection = await self.select_patterns()
                print(f"  Selected patterns: {selection.patterns}")
                print(f"  Scenario type: {selection.scenario_type}")
                log_lines.append(f"[SELECT]\n{selection.model_dump_json(indent=2)}\n")

                scenario = await self.generate_scenario(selection)
                print(f"  Generated {len(scenario.code)} chars with {len(scenario.manifest)} bugs")
                log_lines.append(
                    f"[GENERATE] {len(scenario.code)} chars, {len(scenario.manifest)} bugs\n"
                    f"Manifest: {json.dumps([m.model_dump() for m in scenario.manifest], indent=2)}\n"
                )

                # Step 2: Verify manifest (Sonnet)
                manifest = scenario.manifest
                verified = not skip_verification

                if not skip_verification:
                    print("  Verifying manifest (Sonnet)...")
                    verification = await self.verify_manifest(scenario.code, scenario.manifest)
                    log_lines.append(f"[VERIFY]\n{verification.model_dump_json(indent=2)}\n")

                    if verification.quality == "regenerate":
                        print(f"  Verification failed (attempt {attempt + 1}/{max_attempts})")
                        log_lines.append("[VERIFY] quality=regenerate, retrying...\n")
                        continue

                    if (
                        verification.quality == "needs_correction"
                        and verification.corrected_manifest
                    ):
                        manifest = verification.corrected_manifest
                        print("  Manifest corrected by verifier")

                # Step 3: Run linter (vLLM)
                print("  Running linter (vLLM)...")
                manifest_dicts = [m.model_dump() for m in manifest]
                findings = await self.run_linter(scenario.code)
                print(f"  Linter found {len(findings)} issues")
                log_lines.append(
                    f"[LINT] {len(findings)} findings\n{json.dumps(findings, indent=2)}\n"
                )

                # Step 4: Judge results (Sonnet)
                judge_result = None
                if not skip_judge:
                    print("  Judging results (Sonnet)...")
                    judge_result = await self.judge_findings(
                        scenario.code, manifest_dicts, findings
                    )

                tp_intended, tp_bonus, false_positives, false_negatives, judge_logs = (
                    categorize_judge_results(judge_result, manifest_dicts, findings, skip_judge)
                )
                log_lines.extend(judge_logs)

                # Create scenario name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scenario_name = f"gen_{timestamp}_{selection.scenario_type.replace(' ', '_')[:30]}"
                scenario_name = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in scenario_name
                )

                result = ScenarioResult(
                    name=scenario_name,
                    code=scenario.code,
                    patterns=selection.patterns,
                    manifest=manifest_dicts,
                    verified=verified,
                    bugs_intended=len(manifest_dicts),
                    bugs_detected=len(tp_intended) + len(tp_bonus),
                    tp_intended=len(tp_intended),
                    tp_bonus=tp_bonus,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                )

                log_lines.append(
                    f"[RESULT] TP={result.tp_intended}/{result.bugs_intended}, "
                    f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}, "
                    f"FN={len(result.false_negatives)}\n"
                )
                return result, "\n".join(log_lines)

            except Exception as e:
                print(f"  Error (attempt {attempt + 1}/{max_attempts}): {e}")
                log_lines.append(f"[ERROR] {e}\n")
                continue

        return None, "\n".join(log_lines)

    async def generate_batch(
        self,
        count: int,
        skip_verification: bool = False,
        skip_judge: bool = False,
        run_output: RunOutput | None = None,
    ) -> list[ScenarioResult]:
        """Generate multiple scenarios with optional incremental disk streaming.

        Args:
            count: Number of scenarios to generate.
            skip_verification: Skip manifest verification.
            skip_judge: Skip result judging.
            run_output: Optional output directory for streaming logs to disk.

        Returns:
            List of successful ScenarioResult objects.
        """
        write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()
        writer_task = asyncio.create_task(write_worker(write_queue))
        progress_file = run_output.log.open("a") if run_output else None

        try:
            # Launch all scenarios concurrently (ClaudeCLI rate limiter handles throttling)
            tasks = []
            for i in range(count):
                task = asyncio.create_task(
                    self.generate_one(skip_verification=skip_verification, skip_judge=skip_judge)
                )
                tasks.append((i, task))

            scenarios = []
            completed = 0
            for coro in asyncio.as_completed([t for _, t in tasks]):
                result_tuple = await coro
                scenario, raw_log = result_tuple
                completed += 1

                if scenario:
                    scenarios.append(scenario)
                    msg = (
                        f"[{completed}/{count}] ✓ {scenario.name} "
                        f"TP={scenario.tp_intended}/{scenario.bugs_intended} "
                        f"Bonus={len(scenario.tp_bonus)} FN={len(scenario.false_negatives)}"
                    )
                    print(msg)

                    # Stream to disk
                    if run_output:
                        log_path = run_output.item_file(scenario.name)
                        await write_queue.put((log_path, raw_log))
                else:
                    msg = f"[{completed}/{count}] ✗ Failed to generate scenario"
                    print(msg)

                    # Write failure log too
                    if run_output and raw_log:
                        fail_name = f"failed_{completed}_{datetime.now().strftime('%H%M%S')}"
                        log_path = run_output.item_file(fail_name)
                        await write_queue.put((log_path, raw_log))

                if progress_file:
                    progress_file.write(msg + "\n")
                    progress_file.flush()

            # Signal writer to stop and wait
            await write_queue.put(None)
            await writer_task

            return scenarios
        finally:
            if progress_file:
                progress_file.close()


# =============================================================================
# File Operations
# =============================================================================


def save_scenarios(
    scenarios: list[ScenarioResult],
    run_dir: Path,
    seed: int = 42,
    generation_stats: dict[str, Any] | None = None,
) -> None:
    """Save scenarios to disk in a run directory.

    Args:
        scenarios: List of generated scenarios
        run_dir: Directory for this run (e.g., generated/20260316_003500/)
        seed: Random seed used for generation
        generation_stats: Optional stats about generation (attempts, rejections, etc.)
    """
    scenarios_dir = run_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # Metadata section
    verified_count = sum(1 for s in scenarios if s.verified)
    config: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "seed": seed,
            "total_scenarios": len(scenarios),
            "verified_scenarios": verified_count,
            "unverified_scenarios": len(scenarios) - verified_count,
        },
        "scenarios": {},
    }

    # Add generation stats if provided
    if generation_stats:
        config["metadata"]["generation"] = generation_stats

    for scenario in scenarios:
        # Save code file
        file_path = scenarios_dir / f"{scenario.name}.py"
        file_path.write_text(scenario.code)

        # Convert manifest to expected_patterns format
        expected_patterns: dict[str, int] = {}
        for entry in scenario.manifest:
            pid = entry["pattern_id"]
            expected_patterns[pid] = expected_patterns.get(pid, 0) + 1

        config["scenarios"][scenario.name] = {
            "description": f"Generated scenario with {len(scenario.patterns)} patterns",
            "file": f"scenarios/{file_path.name}",
            "expected_patterns": expected_patterns,
            "bugs": scenario.manifest,
            "verified": scenario.verified,
        }

    config["evaluation"] = {
        "run_all_patterns": True,
        "verbose": True,
        "stop_on_first_failure": False,
    }

    expected_path = run_dir / "expected.yaml"
    with open(expected_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def clear_generated_run(run_dir: Path) -> None:
    """Remove a specific generated run."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"Cleared {run_dir}")


def clear_all_generated(generated_dir: Path) -> None:
    """Remove all generated runs."""
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
        print(f"Cleared all generated runs in {generated_dir}")


def list_saved_runs() -> list[tuple[str, Path, datetime]]:
    """List all saved runs with their modification times."""
    if not GENERATED_DIR.exists():
        return []

    runs = []
    for run_dir in GENERATED_DIR.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            expected_yaml = run_dir / "expected.yaml"
            if expected_yaml.exists():
                mtime = datetime.fromtimestamp(expected_yaml.stat().st_mtime)
                runs.append((run_dir.name, run_dir, mtime))

    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x[2], reverse=True)
    return runs


def load_scenarios_from_disk(
    run_dir: Path, include_unverified: bool = False
) -> list[tuple[str, str, list[dict[str, Any]], bool]]:
    """Load scenarios from a saved run.

    Args:
        run_dir: Directory containing expected.yaml and scenarios/
        include_unverified: If False, skip scenarios that failed verification

    Returns:
        List of (name, code, manifest, verified) tuples
    """
    expected_yaml = run_dir / "expected.yaml"
    if not expected_yaml.exists():
        return []

    with open(expected_yaml) as f:
        config = yaml.safe_load(f)

    scenarios = []
    skipped = 0
    for name, scenario_config in config.get("scenarios", {}).items():
        verified = scenario_config.get("verified", True)

        # Skip unverified scenarios unless explicitly included
        if not verified and not include_unverified:
            skipped += 1
            continue

        # Load code from file
        code_file = run_dir / scenario_config["file"]
        if code_file.exists():
            code = code_file.read_text()
            manifest = scenario_config.get("bugs", [])
            scenarios.append((name, code, manifest, verified))

    if skipped > 0:
        print(f"  Skipped {skipped} unverified scenarios (use --include-unverified to include)")

    return scenarios


async def reeval_existing_run(run_dir: Path, args: argparse.Namespace) -> int:
    """Re-evaluate an existing saved run (lint + judge only, no generation)."""
    include_unverified = getattr(args, "include_unverified", False)
    saved_scenarios = load_scenarios_from_disk(run_dir, include_unverified=include_unverified)
    if not saved_scenarios:
        print(f"Error: No scenarios found in {run_dir}")
        return 1

    print(f"Loaded {len(saved_scenarios)} scenarios from {run_dir}")

    # Create run output for streaming logs
    run_output = RunOutput.create(REPORTS_DIR, f"reeval_{args.id}", items_dirname="scenarios")
    print(f"Output directory: {run_output.run_dir}")
    print(f"Monitor progress: tail -f {run_output.log}")
    run_output.init_log()

    write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(write_worker(write_queue))
    progress_file = run_output.log.open("a")

    try:
        # Create generator just for linter and judge
        generator = ScenarioGenerator(seed=args.seed)

        results: list[ScenarioResult] = []
        total = len(saved_scenarios)

        for i, (name, code, manifest, verified) in enumerate(saved_scenarios):
            print(f"\nEvaluating scenario {i + 1}/{total}: {name}")
            log_lines: list[str] = [f"Scenario: {name}\n{'=' * 60}\n"]

            # Run linter
            print("  Running linter (vLLM)...")
            findings = await generator.run_linter(code)
            print(f"  Linter found {len(findings)} issues")
            log_lines.append(f"[LINT] {len(findings)} findings\n{json.dumps(findings, indent=2)}\n")

            # Judge results
            judge_result = None
            if not args.skip_judge:
                print("  Judging results (Sonnet)...")
                judge_result = await generator.judge_findings(code, manifest, findings)

            tp_intended, tp_bonus, false_positives, false_negatives, judge_logs = (
                categorize_judge_results(judge_result, manifest, findings, args.skip_judge)
            )
            log_lines.extend(judge_logs)

            result = ScenarioResult(
                name=name,
                code=code,
                patterns=[],  # Not stored in saved runs
                manifest=manifest,
                verified=verified,
                bugs_intended=len(manifest),
                bugs_detected=len(tp_intended) + len(tp_bonus),
                tp_intended=len(tp_intended),
                tp_bonus=tp_bonus,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )
            results.append(result)

            msg = (
                f"[{i + 1}/{total}] {name} "
                f"TP={result.tp_intended}/{result.bugs_intended} "
                f"Bonus={len(result.tp_bonus)} FP={len(result.false_positives)}"
            )
            print(
                f"  Result: TP={result.tp_intended}/{result.bugs_intended}, "
                f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}"
            )

            log_lines.append(
                f"[RESULT] TP={result.tp_intended}/{result.bugs_intended}, "
                f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}, "
                f"FN={len(result.false_negatives)}\n"
            )

            # Stream to disk
            log_path = run_output.item_file(name)
            await write_queue.put((log_path, "\n".join(log_lines)))

            progress_file.write(msg + "\n")
            progress_file.flush()

        # Signal writer to stop and wait
        await write_queue.put(None)
        await writer_task
    finally:
        progress_file.close()

    # Print report
    print_report(results)

    # Save JSON report
    if args.output:
        save_json_report(results, args.output)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = REPORTS_DIR / f"integration_eval_{timestamp}.json"
        save_json_report(results, default_output)

    print(f"\nRun logs: {run_output.run_dir}")

    # Summary
    total_intended = sum(s.bugs_intended for s in results)
    total_tp = sum(s.tp_intended for s in results)
    success = total_tp == total_intended

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scenarios: {len(results)}")
    print(f"Detection: {total_tp}/{total_intended} intended bugs")
    if success:
        print("Status: All intended bugs detected")
    else:
        print(f"Status: {total_intended - total_tp} bugs missed")

    return 0 if success else 1


def print_report(scenarios: list[ScenarioResult]) -> None:
    """Print evaluation report to console."""
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp_intended = sum(s.tp_intended for s in scenarios)
    total_tp_bonus = sum(len(s.tp_bonus) for s in scenarios)
    total_fp = sum(len(s.false_positives) for s in scenarios)
    total_fn = sum(len(s.false_negatives) for s in scenarios)

    print("\n" + "=" * 60)
    print("INTEGRATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Bugs intended: {total_intended}")
    print(f"TP-intended (expected bugs found): {total_tp_intended}")
    print(f"TP-bonus (verified extra bugs): {total_tp_bonus}")
    print(f"FP (rejected, not real bugs): {total_fp}")
    print(f"FN (missed bugs): {total_fn}")
    print()

    if total_intended > 0:
        recall = total_tp_intended / total_intended * 100
        print(f"Recall: {recall:.1f}%")

    total_findings = total_tp_intended + total_tp_bonus + total_fp
    if total_findings > 0:
        precision = (total_tp_intended + total_tp_bonus) / total_findings * 100
        print(f"Precision: {precision:.1f}%")

    print("=" * 60)

    # Per-scenario breakdown
    if len(scenarios) > 1:
        print("\nPer-scenario results:")
        for s in scenarios:
            status = "PASS" if s.tp_intended == s.bugs_intended else "FAIL"
            bonus_str = f" +{len(s.tp_bonus)} bonus" if s.tp_bonus else ""
            print(f"  {s.name[:40]}: {s.tp_intended}/{s.bugs_intended} TP{bonus_str} [{status}]")


def save_json_report(scenarios: list[ScenarioResult], output_path: Path) -> None:
    """Save JSON report."""
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp_intended = sum(s.tp_intended for s in scenarios)
    total_tp_bonus = sum(len(s.tp_bonus) for s in scenarios)
    total_fn = sum(len(s.false_negatives) for s in scenarios)

    report = {
        "summary": {
            "scenarios": len(scenarios),
            "total_bugs_intended": total_intended,
            "total_tp_intended": total_tp_intended,
            "total_tp_bonus": total_tp_bonus,
            "total_fn": total_fn,
            "recall": total_tp_intended / total_intended if total_intended > 0 else 0.0,
        },
        "scenarios": [
            {
                "name": s.name,
                "patterns": s.patterns,
                "bugs_intended": s.bugs_intended,
                "tp_intended": s.tp_intended,
                "tp_bonus": s.tp_bonus,
                "false_negatives": s.false_negatives,
                "verified": s.verified,
            }
            for s in scenarios
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Generate and evaluate integration test scenarios."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate integration test scenarios using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and evaluate 10 scenarios (ephemeral - discards after)
  python integration_eval.py --generate-count 10

  # Skip verification/judge for faster (less accurate) evaluation
  python integration_eval.py --generate-count 10 --skip-verification --skip-judge

  # Save with auto-generated ID (timestamp)
  python integration_eval.py --generate-count 10 --save

  # Save with custom ID for regression testing
  python integration_eval.py --generate-count 10 --save --id baseline_v1

  # Re-evaluate existing saved run (no generation)
  python integration_eval.py --id baseline_v1

  # Re-evaluate including unverified scenarios
  python integration_eval.py --id baseline_v1 --include-unverified

  # Force regeneration of existing ID
  python integration_eval.py --generate-count 10 --save --id baseline_v1 --force

  # List all saved runs
  python integration_eval.py --list
""",
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        dest="generate_count",
        help="Number of scenarios to generate (triggers generation mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip manifest verification",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip result judging (use deterministic comparison)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save scenarios to disk for regression tests (default: ephemeral mode)",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Run ID for saved scenarios (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if ID already exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved runs and exit",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        dest="include_unverified",
        help="Include scenarios that failed verification (re-eval mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report path",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        runs = list_saved_runs()
        if not runs:
            print("No saved runs found.")
            print(
                "Generate with: python integration_eval.py --generate-count 10 --save --id my_run"
            )
            return 0
        print("Saved runs:")
        for run_id, _, mtime in runs:
            print(f"  {run_id}  ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        return 0

    # Validate flag combinations
    if args.save and args.generate_count is None and args.id is None:
        print("Error: --save requires --generate-count")
        print("  Example: python integration_eval.py --generate-count 10 --save")
        return 1

    if args.force and (args.generate_count is None or args.id is None):
        print("Error: --force requires both --generate-count and --id")
        print(
            "  Example: python integration_eval.py --generate-count 10 --save --id baseline_v1 --force"
        )
        return 1

    if args.include_unverified and args.generate_count is not None:
        print(
            "Error: --include-unverified only applies to re-eval mode (--id without --generate-count)"
        )
        print("  Example: python integration_eval.py --id baseline_v1 --include-unverified")
        return 1

    # Determine mode: generate or re-eval
    run_id = args.id if args.id else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = GENERATED_DIR / run_id
    id_exists = run_dir.exists() and (run_dir / "expected.yaml").exists()

    # Mode validation
    if args.generate_count is None and args.id is None:
        print("Error: Must specify --generate-count N (generate) or --id X (re-evaluate)")
        print("  Generate: python integration_eval.py --generate-count 10")
        print("  Re-eval:  python integration_eval.py --id baseline_v1")
        return 1

    if args.generate_count is None and args.id:
        # Re-eval mode: ID must exist
        if not id_exists:
            print(f"Error: Run ID '{args.id}' does not exist")
            print("Use --list to see available runs")
            return 1
        # Warn if --save is specified in re-eval mode
        if args.save:
            print("Warning: --save is ignored in re-eval mode (scenarios already saved)")
        print(f"Re-evaluating run '{args.id}'...")
        return asyncio.run(reeval_existing_run(run_dir, args))

    if args.generate_count and args.id and id_exists and not args.force:
        # ID exists but trying to generate - require --force
        print(f"Error: Run ID '{args.id}' already exists")
        print("Use --force to regenerate, or omit --generate-count to re-evaluate")
        return 1

    # Generate mode (args.generate_count is set at this point)
    assert args.generate_count is not None  # for type checker

    # Create generator (uses Claude CLI)
    mode = "persistent (--save)" if args.save else "ephemeral"
    print(f"Mode: {mode}")
    generator = ScenarioGenerator(seed=args.seed)

    # Create run output for streaming logs
    run_output = RunOutput.create(
        REPORTS_DIR, f"generate_{args.generate_count}", items_dirname="scenarios"
    )
    print(f"Output directory: {run_output.run_dir}")
    print(f"Monitor progress: tail -f {run_output.log}")
    run_output.init_log()

    # Generate scenarios
    print(f"\nGenerating {args.generate_count} scenarios (seed={args.seed})...")
    scenarios = asyncio.run(
        generator.generate_batch(
            count=args.generate_count,
            skip_verification=args.skip_verification,
            skip_judge=args.skip_judge,
            run_output=run_output,
        )
    )

    if not scenarios:
        print("\nNo scenarios generated successfully.")
        return 1

    # Print report
    print_report(scenarios)

    # Save if requested
    if args.save:
        save_scenarios(scenarios, run_dir, seed=args.seed)
        print(f"\nSaved {len(scenarios)} scenarios to {run_dir}")
        print(f"Run ID: {run_id}")
        print(f"To re-evaluate: python integration_eval.py --id {run_id}")

    # Save JSON report
    if args.output:
        save_json_report(scenarios, args.output)
    else:
        save_json_report(scenarios, run_output.run_dir / "report.json")

    print(f"\nRun logs: {run_output.run_dir}")

    # Summary
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp = sum(s.tp_intended for s in scenarios)
    success = total_tp == total_intended

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated: {len(scenarios)}/{args.generate_count} scenarios")
    print(f"Detection: {total_tp}/{total_intended} intended bugs")
    if success:
        print("Status: All intended bugs detected")
    else:
        print(f"Status: {total_intended - total_tp} bugs missed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
