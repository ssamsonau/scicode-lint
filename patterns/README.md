# Patterns - Detection Pattern Definitions

Pattern specifications and test data for scicode-lint.

## Overview

This directory contains the **definitions** of what issues to detect and **test data** to validate detection quality.

**Not to be confused with:**
- `evals/` - Evaluation frameworks (code that runs quality checks)
- `tests/` - Unit tests (infrastructure validation)
- `benchmarks/` - Performance measurements

## Directory Structure

```
patterns/
├── ai-training/           # ML training correctness patterns
│   ├── ml-001-scaler-leakage/
│   │   ├── pattern.toml               # Complete pattern definition (detection + tests)
│   │   ├── test_positive/             # Code WITH issues (must detect)
│   │   │   └── scaler_before_split.py
│   │   ├── test_negative/             # Code WITHOUT issues (must NOT detect)
│   │   │   └── scaler_after_split.py
│   │   └── test_context_dependent/    # Ambiguous edge cases
│   │       └── scaler_on_train_val.py
│   ├── ml-002-target-leakage/
│   └── ...
├── ai-inference/          # Inference correctness patterns
├── ai-data/               # Data loading patterns
├── scientific-numerical/  # Numerical precision patterns
├── scientific-performance/ # Performance patterns
└── scientific-reproducibility/ # Reproducibility patterns

Total: 44 patterns
```

## File Roles

Each pattern directory contains exactly 2 things:

### 1. `pattern.toml` - Pattern Definition & Test Cases (Required)

**Purpose**: Complete pattern specification - both detection rules and test cases

**What it contains**:
- **[meta]** - Pattern ID, name, category, severity
- **[detection]** - Detection question (what the LLM looks for), warning message, explanation
- **[tests]** - Test case definitions (positive, negative, context_dependent)

```toml
[meta]
id = "ml-001"
name = "scaler-leakage"
category = "ai-training"
severity = "critical"

[detection]
question = "Is StandardScaler.fit() called on the full dataset before train_test_split()?"
warning_message = "Data leakage: scaler/encoder is fit on full data including test set..."

[[tests.positive]]
file = "test_positive/scaler_before_split.py"
description = "Scaler fit on combined train+test data"
expected_issue = "Data leakage: scaler statistics include test set"
min_confidence = 0.85

[[tests.negative]]
file = "test_negative/scaler_after_split.py"
description = "Correct scaler usage after split"
max_false_positives = 0
```

**What each component sees**:
- **Linter** sees: `[meta]` and `[detection]` sections only
- **Evaluator** sees: `[tests]` section only

### 2. Test Code Files (Required)


**Purpose**: Pure Python code to test detection on (NO hints, NO documentation)

**⚠️ CRITICAL - NO DATA LEAKAGE ALLOWED:**

Test files must contain **ONLY code** - ZERO documentation, ZERO hints, ZERO comments about what's wrong.

**Absolutely forbidden**:
- Module docstrings explaining the issue
- Function docstrings with "ISSUE:", "BUG:", "CORRECT:" markers
- Inline comments like `# Data leakage here` or `# This is wrong`
- Variable names like `buggy_function` or `correct_approach`
- ANY text that tells the LLM what the answer is

**Why**: If test files contain hints, the LLM sees the answer. This invalidates evaluation - we're testing the LLM's detection ability, not its reading comprehension.

All context belongs in `pattern.toml` (detection question and test descriptions), which the linter partially sees (only [detection] section, not [tests]).

#### Positive Tests (Code WITH Issues)

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

**Requirements:**
- Pure code, no documentation
- Realistic, runnable code
- Contains the issue pattern
- 10-50 lines typical

#### Negative Tests (Correct Code)

```python
def normalize_correctly(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()

    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized
```

**Requirements:**
- Pure code, no documentation
- Similar structure to positive (tests pattern detection, not code quality)
- Does NOT contain the issue pattern
- 2-3 different correct implementations

#### Context-Dependent Tests (Ambiguous Cases)

```python
import torch


def fit_on_train_and_val(train_data, val_data, test_data):
    train_val_combined = torch.cat([train_data, val_data])
    mean = train_val_combined.mean()
    std = train_val_combined.std()

    test_normalized = (test_data - mean) / std
    return test_normalized
```

**Requirements:**
- Pure code, no documentation
- Edge case where detection is debatable
- Both "yes" and "no" can be acceptable

## Why No Documentation in Test Files?

**Problem**: If test files have comments like `# BUG: this causes data leakage`, the LLM sees the answer in the code itself. This is data leakage in our evaluation.

**Solution**: Test files are pure code. All context is in pattern.toml:
- **Linter sees**: Code file + `pattern.toml` [detection] section
- **Evaluator sees**: Code file + `pattern.toml` [tests] section
- **Test files see**: Nothing (just pure code)

## Pattern Categories

### ai-training (15 patterns)
Critical ML training correctness issues:
- Data leakage (scaler, target, temporal)
- PyTorch training modes
- Gradient management
- Loss function selection

### ai-inference (3 patterns)
Model inference correctness:
- Missing eval mode
- Missing no_grad context
- Device mismatches

### ai-data (1 pattern)
Data loading issues:
- DataLoader configuration
- dtype mismatches

### scientific-numerical (10 patterns)
Numerical precision and stability:
- Float equality
- NumPy array mutations
- Catastrophic cancellation
- Division by zero

### scientific-performance (11 patterns)
Performance anti-patterns:
- Python loops over NumPy
- Threading on CPU-bound work
- Missing vectorization
- Memory inefficiency

### scientific-reproducibility (4 patterns)
Reproducibility issues:
- Missing random seeds
- CUDA non-determinism
- Hardcoded hyperparameters

## Creating New Patterns

### 1. Create Directory Structure

```bash
mkdir -p patterns/ai-training/ml-999-my-pattern/{positive,negative,context_dependent}
```

### 2. Create `pattern.toml`

```toml
[meta]
id = "ml-999"
name = "my-pattern"
category = "ai-training"
severity = "critical"

[detection]
detection_question = "What specific code pattern should the LLM look for?"
warning_message = "Brief explanation of why this is an issue and how to fix it."
```

### 3. Create `ground_truth.yaml`

```yaml
pattern_id: ml-999-my-pattern
category: ai-training
severity: critical
description: "Brief description of the issue"

positive_cases:
  - file: positive/example.py
    expected_findings:
      - location:
          type: function
          name: problematic_function
          snippet: "key problematic line"
        issue: "What the issue is"
        min_confidence: 0.85

negative_cases:
  - file: negative/correct.py
    max_false_positives: 0

ambiguous_cases:
  - file: context_dependent/edge_case.py
    allow_detection: true
    allow_skip: true
```

### 4. Create Test Files

Write pure Python code (no docstrings, no comments):
- `test_positive/*.py` - 2-3 files with the issue
- `test_negative/*.py` - 2-3 files without the issue
- `test_context_dependent/*.py` - 1 file with edge case

### 4. Validate

```bash
# Run linter
python -m scicode_lint check patterns/ai-training/ml-999-my-pattern/test_positive/*.py --pattern ml-999

# Run evaluation (reads test cases from pattern.toml [tests] section)
python -m evals.run_eval --pattern ml-999-my-pattern
```

## Quality Standards

All patterns must meet:
- **Precision** ≥ 0.90 (minimize false alarms)
- **Recall** ≥ 0.80 (catch most issues)
- **F1 Score** ≥ 0.85 (balanced)
- **Critical patterns**: Precision ≥ 0.95

## Two Evaluation Modes

### 1. Simple Rule-Based (`run_eval.py`)
- Fast: Just checks "did linter detect anything?"
- Good for: Quick iteration, CI/CD

### 2. LLM-as-a-Judge (`run_eval_llm_judge.py`)
- Thorough: Semantic evaluation of correctness
- Checks: Are lines correct? Is reasoning sound?
- Good for: Deep quality assessment

## Current Status

**Complete (8 patterns):**
- ml-001 through ml-008 ✓

**Need test cases (36 patterns):**
- All others have template structure
- Require realistic test files

## File Separation Summary

**pattern.toml has two audiences:**
- **[meta] + [detection]** sections → Read by the linter at runtime
- **[tests]** section → Read by the evaluator only, linter never sees this

This provides logical separation while keeping everything in one file for simplicity.

## See Also

- [evals/](../evals/) - Evaluation frameworks
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute patterns
- [ARCHITECTURE.md](../docs_dev_genai/ARCHITECTURE.md) - Design principles
