# Integration Evaluations (Holdout Set)

Tests generalization - did we overfit to pattern-specific tests?

## Two Approaches

| Approach | Test Data | Use For |
|----------|-----------|---------|
| **Static** | Pre-written scenarios | Holdout - realistic code |
| **Dynamic** | Fresh LLM-generated code | Holdout - fresh code each run |

```bash
# Static integration (pre-written scenarios)
python evals/integration/run_integration_eval.py

# Dynamic integration (fresh generated code - recommended before release)
python evals/integration/dynamic_eval.py --scenarios 10 --bugs 3
```

## Overview

While pattern-specific tests (in `patterns/{category}/{pattern}/`) validate individual patterns in isolation, **integration tests** validate the linter's ability to detect multiple different bugs in realistic code scenarios.

## ⚠️ CRITICAL: No Data Leakage in Test Files

**Integration test files MUST NOT contain any hints about bugs in comments, docstrings, or code:**

❌ **WRONG** - Contains data leakage:
```python
def train_model(model, data):
    # BUG: Missing model.train() call
    # TODO: Should set model to training mode
    for batch in data:
        loss = model(batch)  # Wrong: no train mode
```

✅ **CORRECT** - Clean code:
```python
def train_model(model, data):
    for batch in data:
        loss = model(batch)
        loss.backward()
```

**Why this matters:**
- The LLM sees comments when analyzing code
- Hints like "BUG", "FIXME", "wrong", "missing", "should" give away the answer
- Integration tests must evaluate the linter's detection capability, not its ability to read comments

**Allowed:**
- Module-level docstrings describing the scenario (file purpose)
- Function/class docstrings explaining what the code does
- Normal code comments explaining logic

**NOT allowed in scenario files:**
- Bug annotations ("BUG #1", "FIXME", "TODO")
- Correctness hints ("should", "wrong", "missing", "incorrect")
- Pattern references ("ml-001", "data leakage", "missing train mode")

## Purpose

Integration tests answer questions like:
- Can the linter find multiple different bugs in a single file?
- Do patterns interfere with each other?
- Does the linter work on realistic, complex code?
- What's the overall detection rate across all patterns?

## Structure

```
integration/
  scenarios/              # Test files with multiple injected bugs
    ml_pipeline_complete.py
    pytorch_training_issues.py
    data_preprocessing_bugs.py
  expected_findings.yaml  # Expected detections for each scenario
  run_integration_eval.py # Evaluation script
  README.md              # This file
```

## Test Scenarios

Each scenario is a realistic Python file with **multiple intentionally injected bugs** from different patterns:

1. **`ml_pipeline_complete.py`** - Complete ML pipeline with data leakage, missing mode switches, and reproducibility issues
2. **`pytorch_training_issues.py`** - Training loop with PyTorch-specific bugs
3. **`data_preprocessing_bugs.py`** - Data preprocessing with scikit-learn issues
4. **`repeated_bugs.py`** - Multiple instances of same bugs (tests multi-instance detection)

## Design: One Finding Per Pattern Per File

**Important limitation to understand:**

Each pattern check returns **ONE finding per file**, even if multiple instances of the bug exist.

### Example:

If a file has 3 functions all missing `optimizer.zero_grad()`:

**Current behavior:**
```json
{
  "id": "pt-004",
  "location": {
    "lines": [54, 70, 87],  // All instances listed
    "snippet": "..."
  },
  "explanation": "Missing optimizer.zero_grad() in training loop",
  "confidence": 0.95
}
```
- **1 finding** for pattern pt-004
- **lines array** contains all problematic line numbers [54, 70, 87]

**What we DON'T do:**
- ❌ Create 3 separate findings (one per function)
- ❌ Return pt-004 finding count of 3

### Why This Design?

1. **Architecture**: Each pattern asks one yes/no detection question per file
2. **LLM output**: Returns single structured response with all line numbers
3. **User gets info**: All problematic lines are listed, nothing is hidden
4. **Simplicity**: Cleaner data model (one finding per pattern per file)

### Implication for Tests:

When testing files with multiple instances:
```yaml
# repeated_bugs.py has 3 training functions missing zero_grad
expected_patterns:
  pt-004: 1  # Expect 1 finding (not 3)
  # The finding should contain lines: [54, 70, 87]
```

### Future Enhancement:

Could be extended to return multiple findings per pattern if needed. For now, the single-finding-with-multiple-lines approach provides all necessary information.

## Expected Findings Format

`expected_findings.yaml` defines what the linter should detect:

```yaml
scenarios:
  ml_pipeline_complete:
    description: "Complete ML pipeline with 5 different bugs"
    expected_patterns:
      ml-001-scaler-leakage: 1      # Should find 1 instance
      pt-001-missing-train-mode: 1
      pt-004-missing-zero-grad: 2   # Should find 2 instances
```

## How It Works

### Linter Integration

The evaluation runner uses the **SciCodeLinter directly**:
- Each scenario file is checked with `linter.check_file(path)`
- The linter runs **all patterns concurrently** on each file using `asyncio.gather()`
- Same concurrent execution as production usage
- No mocking - real LLM calls to vLLM server

### Execution Flow

1. Load scenario configurations from `expected_findings.yaml`
2. For each scenario:
   - Read the Python file from `scenarios/`
   - Call `linter.check_file()` (checks all patterns concurrently)
   - Compare findings against expected patterns
   - Calculate metrics (coverage, precision, false positives)
3. Generate overall report

### Performance

- Scenarios run **sequentially** (one file at a time)
- Within each file, all patterns run **concurrently** via the linter
- Typical runtime: ~5-10 seconds per scenario (depends on vLLM server)

## Evaluation Approaches

Like pattern-specific evals, integration tests support **two evaluation methods**:

### 1. Hardcoded Ground Truth (Fast, Deterministic)

**Script**: `run_integration_eval.py`

- Compares pattern IDs against exact expectations
- Fast and deterministic
- Good for regression testing
- Requires exact pattern ID matches

```bash
# All scenarios
python evals/integration/run_integration_eval.py -v

# Specific scenario
python evals/integration/run_integration_eval.py --scenario ml_pipeline_complete -v

# With JSON output
python evals/integration/run_integration_eval.py -v --json results.json
```

### 2. LLM-as-Judge (Flexible, Semantic)

**Script**: `run_integration_eval_llm_judge.py`

- Uses LLM to evaluate semantic correctness
- More flexible - can recognize bugs even if pattern ID differs
- Evaluates explanation quality and reasoning
- Better for development and quality assessment

```bash
# All scenarios with LLM judge
python evals/integration/run_integration_eval_llm_judge.py -v

# Specific scenario
python evals/integration/run_integration_eval_llm_judge.py --scenario ml_pipeline_complete -v

# With JSON output
python evals/integration/run_integration_eval_llm_judge.py -v --json judge_results.json
```

### Which to Use?

- **Hardcoded**: Fast feedback during development
- **LLM Judge**: Quality assurance, evaluating if bugs are detected semantically
- **Both**: Run both before release for comprehensive validation

## Metrics

Integration evaluations track:

- **Coverage**: Percentage of injected bugs detected
- **Precision**: Number of false positives (should be 0)
- **Pattern Distribution**: Which patterns detected what
- **Overall Detection Rate**: Total findings vs expected

### Thresholds

- False positives = 0 (no unexpected findings)
- Min findings met for each scenario

## Success Criteria

A scenario passes if:
1. All expected patterns are detected
2. Zero false positives
3. No crashes or errors during linting

## Adding New Scenarios

1. Create a new Python file in `scenarios/` with multiple bugs
2. Add scenario definition to `expected_findings.yaml`
3. Run evaluation to validate
4. Document which patterns/bugs are included

## Difference from Pattern Tests

| Aspect | Pattern Tests | Integration Tests |
|--------|--------------|-------------------|
| Location | `patterns/{category}/{pattern}/test_*` | `evals/integration/scenarios/` |
| Focus | Single pattern, isolated | Multiple patterns, realistic |
| File Size | Small, minimal | Larger, realistic |
| Bugs | 1 bug per file (usually) | 3-10 bugs per file |
| Purpose | Pattern validation | System validation |

## Dynamic Integration Evaluation

### The Overfitting Problem

Static integration tests use pre-written scenarios. If we tune detection questions based on these results, we risk overfitting - metrics may not generalize to real user code.

### Solution: Dynamic Evaluation

`dynamic_eval.py` generates fresh test code each run:

1. **Generate**: Claude creates Python code with N intentional bugs + a manifest
2. **Lint**: scicode-lint analyzes the generated code (blind - no access to manifest)
3. **Judge**: Claude evaluates results against the manifest

**Data flow (no leakage to linter):**
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Generator  │      │   Linter    │      │    Judge    │
│   (Claude)  │      │ (scicode)   │      │  (Claude)   │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       │ code + manifest    │                    │
       ├───────────────────►│ code only          │
       │                    ├───────────────────►│
       │                    │                    │
       │                    │ detections         │
       │                    ├───────────────────►│
       │                    │                    │
       │ manifest ──────────┼───────────────────►│
       │                    │                    │
       │                    │                    ▼
       │                    │              verdict (TP/FP/FN)
```

**Key design choices:**
- Generator and judge are **separate API calls** (no shared state)
- Linter never sees the manifest - it analyzes code blind
- Judge sees: code, manifest, AND detections (to categorize results)

**Judge categories:**

| Category | Meaning |
|----------|---------|
| **TP-intended** | Encoded bug was detected ✅ |
| **TP-bonus** | Real bug found (not intentionally encoded) 🎁 |
| **FP** | Non-existent bug reported ❌ |
| **FN** | Encoded bug missed ⚠️ |

### Usage

```bash
# Default: 10 scenarios, 3 bugs each
python evals/integration/dynamic_eval.py

# Custom configuration
python evals/integration/dynamic_eval.py --scenarios 10 --bugs 5 --output results.json

# Use Claude CLI instead of API
python evals/integration/dynamic_eval.py --use-cli
```

### When to Use

- **Before releases**: Run dynamic eval to verify metrics generalize
- **After major changes**: Check if improvements hold on fresh code
- **Periodically**: Validate that tuning hasn't caused overfitting

---

## See Also

- [evals/README.md](../README.md) - Pattern-specific evaluations
- [patterns/](../../patterns/) - Pattern definitions
- [benchmarks/](../../benchmarks/) - Performance testing
