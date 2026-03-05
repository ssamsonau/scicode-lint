# Integration Tests Quick Start

## Quick Run

### Hardcoded Ground Truth (Fast)

```bash
# Run all integration tests
python evals/integration/run_integration_eval.py -v

# Run specific scenario
python evals/integration/run_integration_eval.py --scenario ml_pipeline_complete -v

# Generate both markdown and JSON reports
python evals/integration/run_integration_eval.py -v \
  --output evals/integration/report.md \
  --json evals/integration/report.json
```

### LLM-as-Judge (Semantic Evaluation)

```bash
# Run all integration tests with LLM judge
python evals/integration/run_integration_eval_llm_judge.py -v

# Run specific scenario
python evals/integration/run_integration_eval_llm_judge.py --scenario ml_pipeline_complete -v

# Generate reports
python evals/integration/run_integration_eval_llm_judge.py -v \
  --output evals/integration/judge_report.md \
  --json evals/integration/judge_results.json
```

## What Gets Tested

### ml_pipeline_complete.py
Complete ML pipeline with 5 bugs:
- ✗ Data leakage from scaler (ml-001)
- ✗ Target leakage in feature engineering (ml-002)
- ✗ Missing model.train() (pt-001)
- ✗ Missing optimizer.zero_grad() (pt-004)
- ✗ Missing random seed (rep-001)

### pytorch_training_issues.py
PyTorch training with 4 bugs:
- ✗ Missing model.train() in training loop (pt-001)
- ✗ Missing model.eval() in validation (pt-002)
- ✗ Missing optimizer.zero_grad() (pt-004)
- ✗ Inference without model.eval() (pt-007)

### data_preprocessing_bugs.py
Data preprocessing with 4 bugs:
- ✗ Scaler fitted on test data (ml-001)
- ✗ Shuffling time-series data (ml-003)
- ✗ Using future information in features (ml-006)
- ✗ Missing random seed (rep-001)

## Expected Results

### Hardcoded Ground Truth

All scenarios should:
- Find 100% of expected patterns (by exact pattern ID)
- Have 0 false positives
- Exact pattern count matches

### LLM-as-Judge

All scenarios should:
- Recall ≥ 80% (detect at least 80% of bugs)
- Partial credit recall ≥ 90% (including partial detections)
- Judge confirms findings are semantically correct

## Adding New Scenarios

1. Create Python file in `scenarios/` with multiple bugs
2. Add entry to `expected_findings.yaml`
3. Run and validate: `python evals/integration/run_integration_eval.py --scenario your_scenario -v`

## Interpreting Results

### Success
```
✓ All integration tests passed!
Coverage: 100.0% (13/13 bugs detected)
Precision: 100.0%
False Positives: 0
```

### Failure Examples

**Missing bugs:**
```
✗ FAIL - ml_pipeline_complete
Missing patterns: pt-004-missing-zero-grad (expected 1, found 0)
```

**False positives:**
```
✗ FAIL - pytorch_training_issues
False positives: ml-001-scaler-leakage (1 findings)
```

**Coverage below threshold:**
```
Coverage ≥ 90%: ✗ FAIL (76.9%)
```
