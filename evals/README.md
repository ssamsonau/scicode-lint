# Evals - Quality Evaluation Framework

Quality validation for scicode-lint pattern detection.

## Overview

This directory contains tools for evaluating whether pattern detections are **correct** and **helpful**.

**Not to be confused with:**
- `tests/` - Deterministic unit tests (mocked, fast)
- `benchmarks/` - Performance measurements (timing only)
- `patterns/` - Pattern definitions and test data

## Evaluation Types

### 1. Pattern-Specific Evaluations (this directory)

Test individual patterns in isolation using:
- **`run_eval.py`** - Hardcoded ground truth (fast, deterministic)
- **`run_eval_llm_judge.py`** - LLM-as-judge (semantic correctness)

### 2. Integration Evaluations (`integration/`)

Test multiple patterns on realistic code with multiple bugs using:
- **`run_integration_eval.py`** - Hardcoded ground truth (exact pattern ID matching)
- **`run_integration_eval_llm_judge.py`** - LLM-as-judge (semantic bug detection)

## What's in This Directory

### Core Evaluation Scripts

**`run_eval.py`** - Hardcoded ground truth evaluation
- Compares linter output against exact location/snippet expectations
- Rigid but fast and deterministic
- Calculates precision, recall, F1 scores
- Usage: `python evals/run_eval.py --pattern ml-001-scaler-leakage`

**`run_eval_llm_judge.py`** - LLM-as-judge evaluation ✅
- Uses LLM to evaluate semantic correctness
- Flexible, can evaluate explanation quality
- Compares against test case docstrings
- Usage: `python evals/run_eval_llm_judge.py --pattern ml-001-scaler-leakage`

### Supporting Modules

**`metrics.py`** - Precision, recall, F1 calculation
- TP, FP, FN counting
- Aggregation by category and severity
- Threshold checking

**`validators.py`** - Finding validation logic
- Location matching (function/class/method names)
- Snippet comparison
- False positive detection

**`report_generator.py`** - Report generation
- JSON format (machine-readable)
- Markdown format (human-readable)
- Summary statistics

**`test_all_patterns.py`** - Pytest integration
- Parametrized tests for each pattern
- Validates overall metrics
- Checks critical severity precision

### Configuration

**`test_definitions.yaml`** - Evaluation configuration
- Quality thresholds (precision ≥0.90, recall ≥0.80)
- Pattern registry
- Execution settings

## Running Evaluations

### Single Pattern

```bash
# Hardcoded ground truth (exact location matching)
python evals/run_eval.py --pattern ml-001-scaler-leakage

# LLM-as-judge (semantic correctness)
python evals/run_eval_llm_judge.py --pattern ml-001-scaler-leakage

# Compare both approaches
python evals/run_eval.py --pattern ml-001-scaler-leakage && \
python evals/run_eval_llm_judge.py --pattern ml-001-scaler-leakage
```

### All Patterns

```bash
# Hardcoded ground truth - all enabled patterns
python evals/run_eval.py

# LLM-as-judge - all patterns
python evals/run_eval_llm_judge.py

# Generate reports with custom output directory
python evals/run_eval.py --format all --output-dir evals/reports/ground_truth
python evals/run_eval_llm_judge.py --format all --output-dir evals/reports/judge
```

### Via Pytest

```bash
# Run pattern evaluations as tests
pytest evals/test_all_patterns.py -v

# Run specific pattern
pytest evals/test_all_patterns.py -k ml-001
```

## Evaluation Types

### Hardcoded Ground Truth

**Test data location:** `patterns/{category}/{pattern}/`

Each pattern has test files with explicit expectations in `pattern.toml`:

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

**Validation:**
1. Run linter on test file
2. Check detection matches expected location
3. Check confidence ≥ threshold
4. Count TP, FP, FN
5. Calculate metrics

**Pros:**
- Fast and deterministic
- Exact location validation
- Good for regression testing

**Cons:**
- Brittle to prompt changes
- Can't evaluate explanation quality
- Requires exact matches

### LLM-as-Judge ✅

**Test data location:** Same as above, but uses docstrings

Each test file has a docstring describing expected behavior:

```python
"""
Positive test case for pt-004-missing-zero-grad.

This demonstrates a training loop that forgets optimizer.zero_grad(),
causing gradients to accumulate incorrectly across batches.
"""
```

**Validation:**
1. Run linter on test file
2. Extract docstring (expected behavior)
3. Ask LLM judge: "Does output match expected?"
4. Judge returns: yes/no/partial
5. Calculate accuracy

**Pros:**
- Semantic correctness (not just exact matches)
- Can evaluate explanation quality
- More flexible with prompt changes

**Cons:**
- Non-deterministic
- Slower (requires extra LLM calls)
- Needs validation against ground truth

## Metrics

### Pattern-Level Metrics

For each pattern:
- **Precision** = TP / (TP + FP) - minimize false alarms
- **Recall** = TP / (TP + FN) - catch most bugs
- **F1 Score** = harmonic mean of precision and recall

### Overall Metrics

Aggregated across all patterns:
- Overall precision ≥ 0.90
- Overall recall ≥ 0.80
- Critical severity precision ≥ 0.95

### By Category

- ai-training
- ai-inference
- ai-data
- scientific-numerical
- scientific-performance
- scientific-reproducibility

## Output Files

**`reports/`** directory contains:
- `evaluation_report.json` - Full metrics (machine-readable)
- `evaluation_report.md` - Summary report (human-readable)
- Pattern-specific reports

## Integration with Development

### Before Committing Pattern Changes

```bash
# 1. Validate pattern structure
python src/scicode_lint/tools/validate_pattern.py --pattern patterns/ai-training/ml-001

# 2. Run evaluations
python evals/run_eval.py --pattern ml-001

# 3. Check metrics meet thresholds
# - Precision ≥ 0.90
# - Recall ≥ 0.80
# - F1 ≥ 0.85
```

### CI/CD Integration

```yaml
# .github/workflows/ci.yml
- name: Run pattern evaluations
  run: pytest evals/test_all_patterns.py -v
```

## See Also

- [integration/](integration/) - **Multi-pattern integration tests** (NEW)
- [patterns/](../patterns/) - Pattern definitions and test data
- [tests/](../tests/) - Deterministic unit tests
- [benchmarks/](../benchmarks/) - Performance benchmarks
- [ARCHITECTURE.md](../docs_dev_genai/ARCHITECTURE.md) - System design

## Choosing the Right Evaluation

Both evaluation approaches serve different purposes:

- **Hardcoded ground truth**: Fast, deterministic, good for regression testing and CI/CD
- **LLM-as-judge**: Flexible, semantic correctness, good for development and quality assessment

See [EVAL_COMPARISON.md](EVAL_COMPARISON.md) for detailed comparison and when to use each approach.

**Recommended workflow:**
1. Develop patterns with LLM-as-judge feedback (fast iteration)
2. Lock in expectations with ground truth (regression prevention)
3. Run both before release (comprehensive validation)
