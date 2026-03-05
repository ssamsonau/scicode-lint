# Pattern Directory Structure

## Overview

Each pattern follows this exact structure:

```
patterns/ai-training/ml-001-scaler-leakage/
├── pattern.toml              # Detection definition (what to look for)
├── evaluation.yaml           # Test expectations (for eval only)
├── test_positive/            # Code WITH issues (must detect)
│   └── *.py                  # Pure code, NO documentation
├── test_negative/            # Code WITHOUT issues (must not flag)
│   └── *.py                  # Pure code, NO documentation
└── test_context_dependent/   # Edge cases (either OK)
    └── *.py                  # Pure code, NO documentation
```

## File Purposes

### 1. `pattern.toml` - Detection Definition
**Who sees it**: The linter
**What it contains**:
- Pattern metadata (ID, name, category, severity)
- **Detection question** - What the LLM should look for in code
- **Warning message** - What to tell the user if detected
- Explanation of why this is an issue

Example:
```toml
[meta]
id = "ml-001"
name = "scaler-leakage"
category = "ai-training"
severity = "critical"

[detection]
detection_question = "Is StandardScaler.fit() called on the full dataset before train_test_split()?"
warning_message = "Data leakage: scaler/encoder is fit on full data including test set."
```

### 2. `evaluation.yaml` - Test Expectations
**Who sees it**: The evaluator only (NOT the linter)
**What it contains**:
- Pattern ID
- Test case file paths
- Expected behavior descriptions (for LLM-as-a-judge)
- Evaluation metadata

Example:
```yaml
pattern_id: ml-001-scaler-leakage
category: ml-correctness
severity: critical

# Code WITH issues that MUST be detected
positive_cases:
  - file: test_positive/scaler_before_split.py
    expected_findings:
      - issue: "Data leakage: normalization statistics computed from train+test"
        min_confidence: 0.85

# Correct code that must NOT be flagged
negative_cases:
  - file: test_negative/scaler_after_split.py
    max_false_positives: 0

# Edge cases where either outcome is acceptable
context_dependent_cases:
  - file: test_context_dependent/scaler_on_train_val.py
    allow_detection: true
    allow_skip: true
```

### 3. Test Code Files
**Who sees them**: The linter
**What they contain**: Pure Python code with ZERO documentation

**CRITICAL RULE**: Test files must contain ONLY code - no docstrings, no comments explaining what's wrong, no hints.

#### test_positive/ - Code WITH Issues
```python
import torch


def normalize_with_test_stats(train_data, test_data):
    all_data = torch.cat([train_data, test_data])
    mean = all_data.mean()
    std = all_data.std()

    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized
```

#### test_negative/ - Correct Code
```python
def normalize_correctly(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()

    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized
```

#### test_context_dependent/ - Edge Cases
```python
import torch


def fit_on_train_and_val(train_data, val_data, test_data):
    train_val_combined = torch.cat([train_data, val_data])
    mean = train_val_combined.mean()
    std = train_val_combined.std()

    test_normalized = (test_data - mean) / std
    return test_normalized
```

## Why This Structure?

### Separation of Concerns
- **`pattern.toml`**: What the linter looks for (detection logic)
- **`evaluation.yaml`**: What the evaluator expects (test metadata)
- **Test files**: Pure code to analyze (no hints)

### No Data Leakage in Evaluation
If test files had comments like `# ISSUE: this causes data leakage`, the LLM would see the answer. This invalidates the evaluation.

**Solution**: Test files are pure code. All context is in separate files that the linter doesn't see.

### Clear Naming
- `test_positive/` - clearly "test files with issues"
- `test_negative/` - clearly "test files without issues"
- `test_context_dependent/` - clearly "test files that are edge cases"

## What Each Component Sees

```
Linter sees:
  ✓ pattern.toml (detection question)
  ✓ test_*.py files (code to analyze)
  ✗ evaluation.yaml (evaluation metadata)

Evaluator sees:
  ✓ evaluation.yaml (expected behavior)
  ✓ test_*.py files (to compare against expectations)
  ✗ pattern.toml (not needed for eval)

Test files see:
  ✗ Nothing (pure code, zero documentation)
```

## Terminology

- **Issue** (not "bug") - Scientific/logical correctness problems
- **Context-dependent** (not "ambiguous") - Edge cases where detection depends on context
- **Code with issues** (not "buggy code") - More precise terminology

## Migration from Old Structure

Old structure (deprecated):
```
ml-001-scaler-leakage/
├── pattern.toml
├── ground_truth.yaml     ← renamed to evaluation.yaml
├── positive/              ← renamed to test_positive/
├── negative/              ← renamed to test_negative/
└── context_dependent/     ← renamed to test_context_dependent/
```

All references to "bugs", "ambiguous", and old directory names have been updated throughout the codebase.
