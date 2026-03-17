# Integration Evaluations

Automated integration testing with LLM-generated scenarios. No human-in-the-loop.

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. GENERATE (Sonnet)                                       │
│     - Select 2-3 compatible patterns                        │
│     - Generate cohesive code with those bugs                │
│     - Output: code + manifest                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. VERIFY (Sonnet)                                          │
│     - Review: "Does code actually contain these bugs?"      │
│     - Correct manifest if needed                            │
│     - Flag for regeneration if bugs missing                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. LINT (vLLM)                                             │
│     - Run scicode-lint on generated code                    │
│     - Get list of findings                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. JUDGE (Sonnet)                                           │
│     - Categorize each finding: TP-intended / TP-bonus / FP  │
│     - Identify missed bugs (FN)                             │
│     - Calculate precision and recall                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
              ┌──────────────┴──────────────┐
              │                             │
         [--save]                      [default]
              │                             │
    Save to generated/<id>/         Discard temp files
                                    Print report only
```

## Data Flow

All Sonnet calls (steps 1, 2, 4) use `dev_lib.ClaudeCLI.arun_json()` for async execution with structured JSON parsing.

| Step | Input | Output |
|------|-------|--------|
| Generate | Pattern catalog | `GeneratedScenario` (code + manifest) |
| Verify | Scenario | Updated manifest (corrected lines) |
| Lint | Code | Findings list |
| Judge | Findings + manifest | `ScenarioResult` (categorized) |

**Disk streaming**: Each scenario's raw Claude outputs (select, generate, verify, judge) are written to disk incrementally via `dev_lib.RunOutput` + async write queue. If the pipeline crashes mid-run, completed scenario logs are preserved in `reports/<timestamp>/scenarios/`.

**On `--save`**: Final results also written to `generated/<run-id>/`:
- `scenarios/*.py` - Generated code files
- `expected.yaml` - Verified manifest + metadata

## Quick Start

```bash
# Full pipeline (default) - generate and evaluate
python evals/integration/integration_eval.py --generate-count 10

# Skip verification/judge (faster, less accurate)
python evals/integration/integration_eval.py --generate-count 10 --skip-verification --skip-judge

# Save with auto-generated ID (timestamp)
python evals/integration/integration_eval.py --generate-count 10 --save

# Save with custom ID for regression testing
python evals/integration/integration_eval.py --generate-count 10 --save --id baseline_v1

# Re-evaluate existing saved run (no generation)
python evals/integration/integration_eval.py --id baseline_v1

# Re-evaluate including scenarios that failed verification
python evals/integration/integration_eval.py --id baseline_v1 --include-unverified

# Force regeneration of existing ID
python evals/integration/integration_eval.py --generate-count 10 --save --id baseline_v1 --force

# List all saved runs
python evals/integration/integration_eval.py --list
```

## Verification & Retry

When Sonnet verifies a generated scenario, it returns one of:

| Quality | Action |
|---------|--------|
| `good` | Proceed with original manifest |
| `needs_correction` | Use corrected manifest (fix line numbers) |
| `regenerate` | Retry generation (up to 3 attempts) |

If all 3 attempts fail, the scenario is skipped (not saved).

**Re-evaluation**: By default, unverified scenarios are skipped. Use `--include-unverified` to include them.

## Saved Run Metadata

Each saved run includes metadata in `expected.yaml`:

```yaml
metadata:
  created_at: "2026-03-16T01:30:00"
  seed: 42
  total_scenarios: 10
  verified_scenarios: 9
  unverified_scenarios: 1

scenarios:
  scenario_name:
    verified: true  # false if failed verification
    ...
```

## Directory Structure

```
integration/
  generated/                   # Saved runs by ID (--save)
    <run-id>/
      scenarios/*.py           # Generated code files
      expected.yaml            # Verified manifest
  reports/                     # Streamed run logs (gitignored)
    <timestamp>_<scope>/
      summary.md               # Not yet used (placeholder)
      progress.log             # One line per scenario
      scenarios/*.log          # Raw Claude outputs per scenario
      report.json              # JSON report
  integration_eval.py          # Full pipeline (generate + evaluate)
  README.md
  DESIGN_NOTES.md
```

**Git behavior:**
- Named IDs (`--id baseline_v1`) → committed
- Timestamp IDs (auto-generated) → gitignored

## Finding Categories

The Sonnet judge reviews each linter finding against the code:

| Category | Meaning |
|----------|---------|
| **TP-intended** | Finding matches a bug in the manifest |
| **TP-bonus** | Verified: real bug NOT in manifest (bonus find) |
| **FP** | Rejected: not a real bug (false positive) |
| **FN** | Manifest bug not detected (missed) |

## Output Example

```
============================================================
INTEGRATION EVALUATION REPORT
============================================================
Scenarios: 10
Bugs intended: 25
TP-intended (expected bugs found): 22
TP-bonus (verified extra bugs): 8
FP (rejected, not real bugs): 2
FN (missed bugs): 3

Recall: 88.0%
Precision: 93.8%
============================================================
```

## Design: One Finding Per Pattern Per File

Each pattern returns **ONE finding per file**, even if multiple instances exist.

If a file has 3 functions missing `optimizer.zero_grad()`:
- **1 finding** for pattern pt-004
- **lines array** contains all line numbers [54, 70, 87]

See [DESIGN_NOTES.md](DESIGN_NOTES.md) for rationale.

## When to Use

- **Before releases**: Verify detection metrics generalize
- **After major changes**: Check improvements hold on fresh code
- **Periodically**: Validate tuning hasn't caused overfitting

## See Also

- [evals/README.md](../README.md) - Pattern-specific evaluations
- [patterns/](../../src/scicode_lint/patterns/) - Pattern definitions
