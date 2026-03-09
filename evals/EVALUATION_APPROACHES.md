# Evaluation Approaches Comparison

Understanding when to use comprehensive vs quick evaluation.

## Quick Reference

| Script | Command | When to Use |
|--------|---------|-------------|
| **`run_eval.py`** | `python evals/run_eval.py` | Development, quality assessment (comprehensive) |
| **`run_eval.py --skip-judge`** | `python evals/run_eval.py --skip-judge` | Fast regression tests (no judge) |

```bash
# Comprehensive evaluation (recommended for development)
python evals/run_eval.py --pattern ml-001-scaler-leakage

# Quick evaluation (no judge LLM)
python evals/run_eval.py --skip-judge --pattern ml-001-scaler-leakage
```

## Two Evaluation Methods

### 1. Comprehensive Evaluation (`run_eval.py`)

**What it does:**
- Runs LLM judge for semantic correctness
- Also computes direct metrics (location/confidence matching)
- Calculates alignment metrics to identify divergences
- All in a single pass

**Output includes:**
- Judge verdicts (yes/no/partial) for each test
- Direct metrics pass/fail for each test
- Alignment metrics (semantic alignment, quality issue rate, strictness rate)
- Divergent cases highlighted for investigation

**When to use:**
- During development for comprehensive feedback
- When refining patterns
- Before release for quality assessment

### 2. Quick Evaluation (`run_eval.py --skip-judge`)

**What it does:**
- Compares linter output against **exact** expectations in `pattern.toml`
- Checks if detection matched expected location (function/class/line)
- Validates confidence thresholds
- Calculates precision, recall, F1 scores

**No judge LLM** - compares linter output against ground truth deterministically.

**When to use:**
- Regression testing
- Quick validation during development

## What Each Approach Evaluates

### Direct Metrics (both scripts)

Uses hardcoded expectations from `pattern.toml`:

```toml
[[tests.positive]]
file = "positive/bug_example.py"
expected_issue = "Data leakage from scaler"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"
name = "normalize_with_test_stats"
snippet = "all_data = torch.cat([train_data, test_data])"
```

**Logic:**
- ✅ Detection at correct location with sufficient confidence → Pass
- ❌ Detection at wrong location or no detection → Fail

### LLM Judge (comprehensive only)

Uses description from `pattern.toml` (NOT file docstrings - avoids data leakage):

```toml
[[tests.positive]]
file = "test_positive/scaler_before_split.py"
description = "Fitting a scaler on train+test combined before splitting"
expected_issue = "Data leakage inflates model performance"
```

**Logic:**
- Judge compares pattern.toml description vs linter output
- Returns verdict based on **semantic correctness**
- Allows for different wording if intent matches

## Comparison Table

| Aspect | `run_eval.py` | `run_eval.py --skip-judge` |
|--------|---------------|---------------------|
| **Judge LLM** | Yes (semantic evaluation) | No (ground truth only) |
| **Linter LLM** | Yes (vLLM) | Yes (vLLM) |
| **Speed** | Slower (extra judge calls) | Faster |
| **Metrics** | Judge + Direct + Alignment | Direct only |
| **Use Case** | Development, quality | Regression testing |
| **Deterministic** | No | Yes |

## Alignment Metrics (Comprehensive Only)

The `run_eval.py` script computes alignment between direct metrics and LLM judge:

| Metric | Meaning |
|--------|---------|
| **Semantic Alignment** | % of cases where both approaches agree |
| **Quality Issue Rate** | % where direct passes but judge fails |
| **Ground Truth Strictness Rate** | % where direct fails but judge passes |

### Interpreting Divergences

| Direct | Judge | Problem | Fix |
|--------|-------|---------|-----|
| ✅ Pass | ❌ No | Right location, wrong explanation | Improve detection prompts |
| ❌ Fail | ✅ Yes | Ground truth too rigid | Relax `pattern.toml` expectations |

## Recommended Workflow

1. **Development**: Run `run_eval.py` for comprehensive feedback
2. **Review**: Check alignment metrics for quality issues
3. **Regression**: Use `run_eval.py --skip-judge` for fast checks
4. **Release**: Run dynamic integration evaluation

```bash
# Development iteration
python evals/run_eval.py --pattern ml-001-scaler-leakage

# Fast regression check
python evals/run_eval.py --skip-judge
```
