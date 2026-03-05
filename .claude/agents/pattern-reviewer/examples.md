# Pattern Reviewer Agent - Examples

**⚠️ Prerequisites:** All examples require [Claude Code CLI](https://github.com/anthropics/claude-code). This agent is specifically designed for Claude Code and will not work with other interfaces.

---

## Example 1: Review a Single Pattern

### Request
```
Review the ml-001-scaler-leakage pattern comprehensively
```

### Expected Output
```markdown
## Pattern: ml-001 - scaler-leakage

### Summary
- Category: ai-training
- Severity: critical
- Status: ⚠️ Needs Improvement

### Strengths
- Clear detection question focused on scaler fitting order
- Critical severity is appropriate for data leakage
- Good explanation of why this matters in research context

### Issues Found

1. **Description field is truncated**
   - Location: pattern.toml [meta.description]
   - Current: "Is there a scaler, normalizer, or encoder (e"
   - Suggestion: Complete the description sentence

2. **Missing code_patterns field**
   - Location: pattern.toml [detection]
   - Suggestion: Add regex patterns like:
     ```toml
     code_patterns = [
       "fit_transform.*before.*split",
       "StandardScaler\\(\\).*fit.*X(?!_train)"
     ]
     ```

3. **Test case metadata incomplete**
   - Location: pattern.toml [tests.positive]
   - Current: Generic "example.py" placeholder
   - Suggestion: Update to reference actual test files:
     ```toml
     [[tests.positive]]
     file = "positive/scaler_before_split.py"
     description = "Scaler fit on combined train+test data"
     ```

### Missing Test Cases

**Positive cases to add:**
- LabelEncoder fit on full dataset
- Pipeline with fit_transform before split
- MinMaxScaler on full feature set

**Negative cases to add:**
- Pipeline usage (correct approach)
- Scaler fit only on training data in cross-validation
- Separate scalers for different feature groups

**Context-dependent cases:**
- Fitting on train+validation (but not test)
- Scaler fit on full data for visualization (not training)

### Recommended Actions

1. Fix truncated description in pattern.toml
2. Add actual test file references to TOML
3. Create 2-3 more positive test cases
4. Create 2-3 negative test cases showing correct approaches
5. Add 1-2 context-dependent cases

### New Test Case Suggestions

**Positive case: LabelEncoder leakage**
```python
# test_positive/label_encoder_before_split.py
from sklearn.preprocessing import LabelEncoder

def encode_before_split(data, labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    train_data = data[:800]
    test_data = data[800:]

    return train_data, test_data
```

**Note**: All context about WHY this is wrong goes in `pattern.toml` [tests] section, NOT in the test file.

**Negative case: Pipeline approach**
```python
# test_negative/correct_pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def correct_pipeline_approach(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    scores = cross_val_score(pipeline, X, y, cv=5)

    return scores
```

**Note**: All context about WHY this is correct goes in `pattern.toml` [tests] section, NOT in the test file.
```

---

## Example 2: Review All Patterns in a Category

### Request
```
Review all patterns in the ai-training category and identify the top 3 priorities for improvement
```

### Expected Output
```markdown
## AI-Training Category Review

### Overview
- Total patterns: 15
- Critical severity: 8
- High severity: 5
- Medium severity: 2

### Summary by Status
- ✅ Good (7 patterns)
- ⚠️ Needs Improvement (6 patterns)
- ❌ Issues Found (2 patterns)

### Top 3 Priorities for Improvement

#### 1. ml-001-scaler-leakage (Priority: HIGH)
**Issues:**
- Truncated description field
- Missing actual test file references
- Only 1 positive test case (need 3-4)
- No negative test cases showing Pipeline approach

**Impact:** Critical severity pattern with incomplete test coverage

**Estimated effort:** 2-3 hours to add test cases and fix metadata

---

#### 2. ml-003-cross-validation-leakage (Priority: HIGH)
**Issues:**
- Detection question too broad (asks multiple things)
- No context-dependent cases for train+val scenarios
- Warning message not specific enough

**Impact:** Common mistake but detection may have false positives

**Estimated effort:** 1-2 hours to refine question and add edge cases

---

#### 3. ml-007-temporal-leakage (Priority: MEDIUM)
**Issues:**
- Only 1 negative test case
- Missing variations (time series vs sequential data)
- No references to time series best practices

**Impact:** Important pattern but narrow test coverage

**Estimated effort:** 2 hours to add test variations

### Patterns Needing Minor Updates (4)
- ml-002: Add tags for related patterns
- ml-004: Fix typo in warning message
- ml-005: Add reference links
- ml-008: Update quality targets

### Patterns in Good Shape (8)
All other patterns have complete metadata and comprehensive test coverage.
```

---

## Example 3: Create New Test Cases

### Request
```
Create 3 new negative test cases for ml-001-scaler-leakage that demonstrate correct approaches to avoid false positives
```

### Expected Output
```markdown
## New Negative Test Cases for ml-001-scaler-leakage

### 1. Pipeline with Cross-Validation (Recommended)

**File:** `test_negative/correct_pipeline_cv.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_with_pipeline(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    scores = cross_val_score(pipeline, X, y, cv=5)

    return np.mean(scores)
```

**Note**: Test files contain ONLY code. Context about why this is correct goes in `pattern.toml` description field.

---

### 2. Manual Train-Test Split with Correct Scaler Usage

**File:** `test_negative/scaler_after_split.py`

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def normalize_after_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
```

---

### 3. Feature Engineering with Separate Scalers

**File:** `test_negative/separate_feature_scalers.py`

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def scale_feature_groups_correctly(X, y, numeric_cols, categorical_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_numeric = X_train[:, numeric_cols]
    X_train_categorical = X_train[:, categorical_cols]
    X_test_numeric = X_test[:, numeric_cols]
    X_test_categorical = X_test[:, categorical_cols]

    numeric_scaler = StandardScaler()
    categorical_scaler = MinMaxScaler()

    X_train_numeric_scaled = numeric_scaler.fit_transform(X_train_numeric)
    X_train_categorical_scaled = categorical_scaler.fit_transform(X_train_categorical)

    X_test_numeric_scaled = numeric_scaler.transform(X_test_numeric)
    X_test_categorical_scaled = categorical_scaler.transform(X_test_categorical)

    X_train_final = np.hstack([X_train_numeric_scaled, X_train_categorical_scaled])
    X_test_final = np.hstack([X_test_numeric_scaled, X_test_categorical_scaled])

    return X_train_final, X_test_final, y_train, y_test
```

---

### Summary

**Created 3 negative test cases:**
1. ✅ Pipeline with cross-validation (prevents false positive on correct CV usage)
2. ✅ Manual train-test split with correct scaler (prevents false positive on proper split-then-fit)
3. ✅ Multiple scalers for feature groups (prevents false positive on complex feature engineering)

**Important**: All test files contain ONLY executable code with NO docstrings or comments. Context about why these are correct belongs in `pattern.toml` [tests] section.

**Next steps:**
1. Save these files to `patterns/ai-training/ml-001-scaler-leakage/test_negative/`
2. Update `pattern.toml` to reference these test cases
3. Run evaluation: `python evals/run_eval.py --pattern ml-001-scaler-leakage`
4. Verify precision/recall meet targets
```

---

## Example 4: Find Patterns Needing Improvement

### Request
```
Which patterns have unclear detection questions or missing test cases? Give me the top 5.
```

### Expected Output
```markdown
## Top 5 Patterns Needing Improvement

### 1. ml-003-cross-validation-leakage
**Category:** ai-training | **Severity:** critical

**Issues:**
- ❌ Detection question asks multiple things at once:
  - "Is data preprocessed before CV?"
  - "Is there data leakage?"
  - "Are preprocessing steps inside or outside CV?"

**Recommendation:** Split into focused yes/no question:
```toml
question = """
Is there preprocessing (scaling, encoding, feature selection) that happens
on the full dataset BEFORE cross-validation splitting?
"""
```

**Missing test cases:**
- Need context-dependent case for train+val fitting
- Need negative case showing ColumnTransformer in Pipeline

---

### 2. pt-003-gradient-accumulation-reset
**Category:** ai-training | **Severity:** high

**Issues:**
- ⚠️ Only 1 positive test case (need 3-4 variations)
- ⚠️ No negative test showing correct `optimizer.zero_grad()` usage
- ⚠️ Detection question doesn't mention `set_to_none=True` optimization

**Recommendation:** Add test cases for:
- Accumulation with proper resets
- `zero_grad()` vs `zero_grad(set_to_none=True)`
- Multi-step accumulation patterns

---

### 3. num-002-dtype-overflow
**Category:** scientific-numerical | **Severity:** high

**Issues:**
- ❌ Detection question too vague: "Could there be overflow?"
- ⚠️ No context-dependent cases (some operations safe despite large numbers)
- ⚠️ Warning message doesn't suggest dtype upgrade

**Recommendation:** Make question specific:
```toml
question = """
Are there arithmetic operations (multiplication, exponentiation, accumulation)
on integer types (int8, int16, int32) that could exceed the dtype's maximum value?
"""
```

---

### 4. rep-001-missing-random-seeds
**Category:** scientific-reproducibility | **Severity:** high

**Issues:**
- ⚠️ Detection question only asks about torch/numpy seeds
- ⚠️ Missing: random, CUDA, dataloader seeds
- ⚠️ No test case for incomplete seeding (some libs seeded, others not)

**Recommendation:** Expand question to cover all sources:
- Python's `random` module
- `torch.cuda.manual_seed_all()`
- DataLoader worker seeding
- Third-party libraries (e.g., transformers)

---

### 5. perf-002-unnecessary-copies
**Category:** scientific-performance | **Severity:** medium

**Issues:**
- ⚠️ Detection question mentions "unnecessary" but doesn't define criteria
- ⚠️ Many context-dependent scenarios (defensive copies often good)
- ❌ No negative test showing when copies ARE necessary

**Recommendation:** Add context to question:
```toml
question = """
Are there tensor or array copies (.copy(), .clone(), .detach().clone()) in
tight loops or repeated operations where views or in-place operations could
be used instead?
"""
```

Add context-dependent cases for:
- Defensive copies for safety
- Copies to prevent aliasing bugs
- Necessary detach+clone for gradient isolation

---

### Summary Table

| Pattern | Issue Type | Priority | Est. Effort |
|---------|-----------|----------|-------------|
| ml-003 | Unclear question | HIGH | 1-2 hours |
| pt-003 | Missing tests | HIGH | 2-3 hours |
| num-002 | Vague question | MEDIUM | 1 hour |
| rep-001 | Incomplete scope | MEDIUM | 2 hours |
| perf-002 | Missing context | LOW | 1-2 hours |

### Recommended Order
1. ml-003 (critical severity, high impact)
2. pt-003 (high severity, common issue)
3. num-002 (affects detection accuracy)
4. rep-001 (important for reproducibility)
5. perf-002 (lower priority, medium severity)
```

---

## Example 5: Iterative Improvement Based on Eval Results

### Request
```
I ran the evaluation for ml-001 and got precision=0.87 (below target 0.90).
What's causing false positives and how can I improve it?
```

### Expected Output
```markdown
## ml-001 Precision Analysis & Improvement Plan

### Current Status
- **Precision:** 0.87 (Target: 0.90)
- **Gap:** -0.03 (need to reduce false positives)
- **Severity:** Critical (target precision: 0.95)

### Likely Causes of False Positives

#### 1. Overly Broad Detection Question
**Current question:**
> "Is there a scaler, normalizer, or encoder that is fit or fit_transform'd on the full dataset before a train/test split occurs?"

**Problem:** Might trigger on:
- Scalers used for visualization (not training)
- Feature exploration before modeling
- Data validation preprocessing

**Fix:** Add context to question:
```toml
question = """
Is there a scaler, normalizer, or encoder (e.g., StandardScaler, MinMaxScaler,
LabelEncoder) that is fit or fit_transform'd on the full dataset BEFORE a
train/test/validation split, AND this scaled data is subsequently used for
model training or evaluation?

Exclude:
- Preprocessing for visualization only
- Data validation/exploration steps
- Summary statistics computation
"""
```

#### 2. Missing "Legitimate Full-Dataset Scaling" Negative Cases

**Add these negative test cases:**

```python
# test_negative/scaling_for_visualization.py
def visualize_data_distribution(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    plot_distributions(scaled_data)
```

```python
# test_negative/data_validation_scaling.py
def check_data_quality(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    outliers = find_outliers(scaled)

    return outliers
```

**Context in pattern.toml [tests] section:**
```toml
[[tests.negative]]
file = "test_negative/scaling_for_visualization.py"
description = "Scaling for visualization only (not used for training)"

[[tests.negative]]
file = "test_negative/data_validation_scaling.py"
description = "Scaling for data quality checks (not used for training)"
```

#### 3. Context-Dependent: Train+Val Scaling

**Problem:** Current pattern might flag train+validation scaling as leakage

**Add context-dependent case:**
```python
# test_context_dependent/scale_train_and_val.py
def fit_scaler_on_train_val(X_train, X_val, X_test):
    X_train_val = np.vstack([X_train, X_val])
    scaler = StandardScaler()
    scaler.fit(X_train_val)

    return scaler
```

**Context in pattern.toml [tests] section:**
```toml
[[tests.context_dependent]]
file = "test_context_dependent/scale_train_and_val.py"
allow_detection = true
allow_skip = true
description = "Fitting on train+val but not test - debatable if this is leakage"
context_notes = """
Acceptable IF:
- Test set is completely held out
- Used for hyperparameter tuning only
Problematic IF:
- Test set used for model selection
- Reporting validation metrics as final performance
"""
```

### Improvement Plan

**Step 1: Refine Detection Question** (Estimated impact: +0.02 precision)
- Add exclusions for non-training uses
- Make "used for training" explicit requirement

**Step 2: Add Negative Cases** (Estimated impact: +0.01 precision)
- Add 2 negative cases for legitimate full-dataset scaling
- Test visualization and data validation scenarios

**Step 3: Add Context Notes** (Estimated impact: maintain current recall)
- Add context-dependent case for train+val scaling
- Document when this is acceptable vs problematic

**Step 4: Update Warning Message**
- Make warning more specific about training vs other uses
- Add note about when full-dataset scaling is acceptable

### Updated TOML Sections

```toml
[detection]
question = """
Is there a scaler, normalizer, or encoder (e.g., StandardScaler, MinMaxScaler,
LabelEncoder) that is:
1. Fit or fit_transform'd on the full dataset before train/test split, AND
2. The resulting scaled data is used for model training or evaluation?

Do NOT flag if scaling is only used for:
- Visualization or plotting
- Data quality checks
- Exploratory analysis not used in modeling
"""

warning_message = """
Data leakage: Scaler/encoder is fit on full data including test set, and the
scaled data is used for model training. This leaks test set statistics into
the training process, inflating performance metrics.

Fix: Use sklearn.pipeline.Pipeline to ensure preprocessing happens inside
cross-validation folds, or fit scaler only on training data after splitting.

Note: Full-dataset scaling is acceptable for visualization or data validation
if the scaled data is not used for training.
"""

false_positive_risks = [
  "Scaling used only for visualization",
  "Data validation preprocessing not used in training",
  "Separate scaling for different purposes (viz vs training)"
]
```

### Expected Results After Improvements

- **Precision:** 0.90-0.92 (meets target)
- **Recall:** 0.78-0.82 (maintained or slightly reduced)
- **F1:** 0.84-0.87

### Verification Steps

1. Update pattern.toml with refined question
2. Add 2 new negative test cases
3. Add 1 context-dependent case
4. Run evaluation: `python evals/run_eval.py --pattern ml-001-scaler-leakage`
5. Check precision ≥ 0.90
6. If still below target, analyze false positive examples from eval output
```
