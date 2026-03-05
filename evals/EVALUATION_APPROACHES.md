# Evaluation Approaches Comparison

Understanding when to use hardcoded ground truth vs LLM-as-judge evaluation.

## Two Approaches

### 1. Hardcoded Ground Truth (`run_eval.py`)

**What it does:**
- Compares linter output against **exact** expectations in `pattern.toml`
- Checks if detection matched expected location (function/class/line)
- Validates confidence thresholds
- Calculates precision, recall, F1 scores

**Example expectation:**
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

**Evaluation logic:**
- ✅ Detection at `normalize_with_test_stats` with snippet match → True Positive
- ❌ Detection at different function → False Positive
- ❌ No detection → False Negative

### 2. LLM-as-Judge (`run_eval_llm_judge.py`)

**What it does:**
- Extracts expected behavior from test file **docstring**
- Runs linter, gets detection output
- Asks LLM judge: "Does output match expected behavior?"
- Judge returns: yes / no / partial

**Example expectation (from docstring):**
```python
"""
Positive test case for ml-001-scaler-leakage.

This code demonstrates fitting a scaler on train+test combined,
which causes data leakage and inflates model performance.
"""
```

**Evaluation logic:**
- Judge compares docstring (expected) vs linter output (actual)
- Returns verdict based on **semantic correctness**
- Allows for different wording if intent matches

## When to Use Each

### Use Hardcoded Ground Truth When:

✅ **Regression testing** - Prevent unintended changes
- Fixed expectations catch when prompts/models change behavior
- Fast and deterministic (no LLM calls during validation)

✅ **CI/CD pipelines** - Quick validation
- No need for LLM server running
- Predictable pass/fail criteria

✅ **Exact location matters** - Precision is critical
- Need to validate detection points to specific code locations
- Testing location matching logic

✅ **Performance testing** - Measure speed
- Can run without LLM overhead
- Benchmark optimizations

### Use LLM-as-Judge When:

✅ **Semantic correctness** - Intent matters more than exact wording
- Detection is correct but phrased differently
- Explanation quality evaluation

✅ **Flexible evaluation** - Multiple valid answers
- Context-dependent cases with nuance
- Edge cases where exact match is too rigid

✅ **Explanation quality** - Evaluate helpfulness
- Judge can assess if warning message is actionable
- Evaluate clarity and completeness

✅ **Development iteration** - Refining patterns
- Get feedback on detection quality
- Identify areas for improvement

## Comparison Table

| Aspect | Hardcoded Ground Truth | LLM-as-Judge |
|--------|----------------------|--------------|
| **Execution** | Deterministic | Non-deterministic |
| **Speed** | Fast (no LLM calls) | Slower (LLM judge calls) |
| **Flexibility** | Rigid (exact matches) | Flexible (semantic) |
| **Use Case** | Regression, CI/CD | Development, quality |
| **Validation** | Location + snippet | Semantic correctness |
| **Output** | Precision/Recall/F1 | Accuracy (yes/no/partial) |
| **Setup** | Requires TOML specs | Requires docstrings |
| **Evaluation** | Rule-based matching | LLM judgment |

## Recommended Workflow

### Development Phase

1. **Initial pattern creation**
   ```bash
   # Create pattern with test cases
   python -m scicode_lint.tools.new_pattern --id ml-999 --name my-pattern

   # Write test files with docstrings
   # (positive/negative/context_dependent)
   ```

2. **Iterate with LLM-as-judge**
   ```bash
   # Get semantic feedback
   python evals/run_eval_llm_judge.py --pattern ml-999-my-pattern

   # Refine based on judge verdicts
   # - If judge says "no", check why detection failed
   # - If judge says "partial", improve explanation
   ```

3. **Lock in with ground truth**
   ```bash
   # Once satisfied, add exact expectations to pattern.toml
   # This creates regression tests

   # Validate
   python evals/run_eval.py --pattern ml-999-my-pattern
   ```

### Pre-Release Validation

```bash
# Run both evaluations
python evals/run_eval.py  # Must pass (regression)
python evals/run_eval_llm_judge.py  # Should pass (quality)

# Compare results
# - Ground truth: Precision ≥0.90, Recall ≥0.80
# - LLM judge: Accuracy ≥0.85
```

### CI/CD Pipeline

```bash
# Fast validation only
pytest evals/test_all_patterns.py -v

# Or direct ground truth eval
python evals/run_eval.py --format json
```

## Example: Same Pattern, Both Approaches

### Test File: `positive/scaler_before_split.py`

```python
"""
Positive test case for ml-001-scaler-leakage.

This demonstrates fitting a scaler on train+test combined
before splitting, which causes data leakage.
"""

def normalize_with_test_stats(train_data, test_data):
    """
    BUG: Scaler fit on combined data.
    """
    all_data = np.concatenate([train_data, test_data])
    scaler = StandardScaler()
    scaler.fit(all_data)  # BUG: includes test set
    return scaler.transform(train_data), scaler.transform(test_data)
```

### Hardcoded Evaluation

**Expectation in pattern.toml:**
```toml
[[tests.positive]]
file = "positive/scaler_before_split.py"
[tests.positive.expected_location]
type = "function"
name = "normalize_with_test_stats"
snippet = "all_data = np.concatenate"
```

**Linter output:**
```json
{
  "detected": true,
  "location": {
    "type": "function",
    "name": "normalize_with_test_stats",
    "line": 12
  },
  "snippet": "all_data = np.concatenate([train_data, test_data])"
}
```

**Result:** ✅ **True Positive** (exact match)

### LLM-as-Judge Evaluation

**Expected (from docstring):**
```
This demonstrates fitting a scaler on train+test combined
before splitting, which causes data leakage.
```

**Linter output:**
```
Detected: true
Issue: "Data leakage: scaler statistics include test set"
Confidence: 0.92
Explanation: "StandardScaler is fit on combined train+test data..."
```

**Judge prompt:**
```
Does the linter output correctly match the test case's intended behavior?

Consider: Did the linter detect the data leakage bug described?
```

**Judge verdict:** ✅ **"yes"** - Semantically correct

## Both Pass Scenarios

**Scenario 1: Exact match**
- Ground truth: ✅ Location and snippet match
- LLM judge: ✅ "yes" - correct detection

**Scenario 2: Different wording, same meaning**
- Ground truth: ✅ Still matches
- LLM judge: ✅ "yes" - semantic match

## Divergence Scenarios

**Scenario 3: Correct bug, wrong location**
- Ground truth: ❌ Location doesn't match (False Positive)
- LLM judge: ✅ "yes" - bug correctly identified

→ Indicates location matching may be too strict

**Scenario 4: Detected, but wrong bug**
- Ground truth: ❌ Issue doesn't match
- LLM judge: ❌ "no" - wrong bug identified

→ Both agree: detection is incorrect

**Scenario 5: Poor explanation**
- Ground truth: ✅ Detection matched
- LLM judge: 🟡 "partial" - detection correct but unclear

→ Suggests explanation needs improvement

## Interpreting Results

### Both Pass → ✅ High Confidence
Pattern is working correctly both technically and semantically.

### Both Fail → ❌ Needs Fixing
Pattern has fundamental issues with detection.

### Ground Truth Pass, Judge Fail → ⚠️ Quality Issue
- Detection is technically correct (right location)
- But semantically wrong (wrong bug) or unclear explanation
- **Action:** Improve detection question or warning message

### Ground Truth Fail, Judge Pass → ⚠️ Overly Strict
- Detection is semantically correct
- But doesn't match exact location/snippet expectations
- **Action:** Review ground truth expectations (may be too rigid)

## Summary

**Use both approaches together:**
- **Ground truth** ensures technical correctness and prevents regression
- **LLM judge** ensures semantic correctness and explanation quality

**Recommended practice:**
1. Develop with LLM-as-judge (flexible, fast iteration)
2. Lock in with ground truth (regression prevention)
3. Validate both before release (comprehensive quality)
