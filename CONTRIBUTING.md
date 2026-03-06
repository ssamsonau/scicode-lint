# Contributing to scicode-lint

Thank you for your interest in contributing to scicode-lint! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding New Detection Rules](#adding-new-detection-rules)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

## Getting Started

Fork and clone the repository.

## Development Setup

### Install in editable mode

```bash
cd scicode-lint

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
scicode-lint --help
```

### Dependencies

All project dependencies are managed in `pyproject.toml`. When adding dependencies, update `pyproject.toml`.

### Install LLM backend

**vLLM Server**
```bash
# Install vLLM
pip install scicode-lint[vllm-server]

# Start vLLM server
vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic \
    --trust-remote-code --max-model-len 16000
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup.

## How to Contribute

### Types of Contributions

- **Bug reports**: Open an issue with details about the problem
- **Feature requests**: Propose new detection patterns or features
- **Documentation**: Improve READMEs, docstrings, or guides
- **Code**: Fix bugs, add features, or improve existing code
- **Detection patterns**: Add new scientific code anti-patterns

## Adding New Detection Patterns

Patterns are the heart of scicode-lint. Each pattern detects a specific type of bug or anti-pattern in scientific Python code. This guide walks you through creating a new pattern from start to finish.

### ⚠️ Critical: Two-Tier LLM Strategy

scicode-lint uses **different models for different purposes**:

**Runtime (Local 12B model)**: Bug detection runs on a constrained-capacity local LLM (Gemma 3 12B). This enables privacy, no API costs, and fast inference.

**Development (SOTA cloud models)**: Pattern development should use the best available reasoning models (Claude, etc.) to ensure high-quality detection questions that the local model can execute reliably.

Detection questions MUST be written FOR constrained-capacity models:
- **Ask about the BUG directly** - "Is X MISSING?" not "Does it have X?"
- **YES = BUG, NO = OK** - The answer directly indicates bug presence
- **Simple and direct** - One search, one check, binary answer
- **No complex reasoning** - If the model needs to "think about it," simplify

**💡 Use the Pattern Reviewer agent** (SOTA model) to review and improve patterns before submission.

See [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) for the full rationale.

**💡 Development tool available:** The [Pattern Reviewer agent](.claude/agents/pattern-reviewer/) can help review, improve, and create test cases for patterns. Requires [Claude Code CLI](https://github.com/anthropics/claude-code). See [docs_dev_genai/TOOLS.md](docs_dev_genai/TOOLS.md) for details.

### Overview

Each pattern is a self-contained directory containing:
- `pattern.toml` - Complete pattern definition (metadata, detection logic, test specs)
- `test_positive/` - Example code that SHOULD trigger detection (bugs)
- `test_negative/` - Example code that should NOT trigger detection (correct code)
- `test_context_dependent/` - Edge cases where either outcome is acceptable

### Pattern Categories

Choose the appropriate category for your pattern:

| Category | Description | Example Patterns |
|----------|-------------|------------------|
| `ai-training` | ML training bugs (data leakage, gradients, loops) | Scaler fit before split, missing .train() |
| `ai-inference` | ML inference issues (eval mode, no_grad, device) | Inference without .eval(), missing torch.no_grad() |
| `ai-data` | Data loading problems (DataLoader config) | DataLoader num_workers=0 |
| `scientific-numerical` | Numerical computing errors (float, NaN, broadcasting) | Float equality comparison, integer overflow |
| `scientific-reproducibility` | Reproducibility issues (seeds, determinism) | Missing random seed, CUDA non-determinism |
| `scientific-performance` | Performance anti-patterns (loops, parallelization) | Inefficient loops, GIL issues |

### Step-by-Step Guide

#### Step 1: Identify the Pattern

Before creating a pattern, clearly define:
1. **What bug/anti-pattern are you detecting?** Be specific
2. **Why does this matter?** What's the research impact?
3. **How can it be fixed?** Provide actionable guidance
4. **What are edge cases?** When is it ambiguous?

**Example**: Data leakage from fitting a scaler on the full dataset before train/test split inflates model performance metrics and leads to wrong scientific conclusions.

#### Step 2: Create Pattern Scaffold

Use the pattern creation tool to generate the initial structure:

```bash
python -m scicode_lint.tools.new_pattern \
    --id ml-050 \
    --name temporal-split-leakage \
    --category ai-training \
    --severity critical
```

This creates:
```
patterns/ai-training/ml-050-temporal-split-leakage/
├── pattern.toml           # Template to fill out
├── test_positive/
│   └── example_positive.py
├── test_negative/
│   └── example_negative.py
└── test_context_dependent/
    └── example_context.py
```

#### Step 3: Edit pattern.toml

Open `patterns/ai-training/ml-050-temporal-split-leakage/pattern.toml` and fill in the template:

**[meta] section** - Pattern metadata:
```toml
[meta]
id = "ml-050"
name = "temporal-split-leakage"
category = "ai-training"
severity = "critical"
version = "1.0.0"
created = "2026-03-03"
updated = "2026-03-03"
author = "scicode-lint"

description = """
Detects time-series data shuffled before train/test split, causing temporal leakage.
"""

explanation = """
Time-series data requires chronological splitting to prevent future data from leaking
into training. Using random_state or shuffle=True breaks temporal ordering, allowing
the model to learn from future data it shouldn't have access to.

Fix: Use TimeSeriesSplit or ensure data is split chronologically without shuffling.
"""

tags = ["data-leakage", "time-series", "temporal"]
related_patterns = ["ml-001", "ml-003"]
references = [
    "https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split"
]
```

**[detection] section** - LLM detection logic:

**⚠️ Detection questions must be ultra-simple for constrained-capacity models:**

```toml
[detection]
# Use this format: Find X, check for Y, map to YES/NO
question = """
Find train_test_split() on time-series data (dates/timestamps in column names).
Does it use shuffle=True?

YES = train_test_split with shuffle=True on time-series (BUG)
NO = shuffle=False, OR no train_test_split, OR not time-series data
"""

warning_message = """
Temporal leakage: time-series data shuffled before split. Use TimeSeriesSplit or shuffle=False.
"""

min_confidence = 0.85
```

**Detection question principles (for 12B models):**
- ✅ Ask about BUG directly: "Is X MISSING?" or "Is X done BEFORE Y?"
- ✅ YES = BUG: The question should make YES mean "bug found"
- ✅ One search, one check: Find X, check for Y
- ✅ Binary decision: No "consider" or "might be"
- ❌ Contradictory mapping: "Does it have X?" + "YES = without X (BUG)"
- ❌ Complex reasoning: "Is there a pattern where X and then Y unless Z..."
- ❌ Vague escape clauses: "Context suggests..." or "might be handled elsewhere"

#### Step 4: Add Test Cases

Create Python files demonstrating the bug and correct code:

**test_positive/shuffled_timeseries.py** - Code that MUST trigger detection:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_timeseries_data(df):
    """Time-series split with shuffle - INCORRECT"""
    X = df.drop('target', axis=1)
    y = df['target']

    # BUG: shuffle=True on time-series data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    return X_train, X_test, y_train, y_test
```

**test_negative/chronological_split.py** - Code that should NOT trigger:
```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def prepare_timeseries_data(df):
    """Correct time-series split - chronological order maintained"""
    X = df.drop('target', axis=1)
    y = df['target']

    # CORRECT: TimeSeriesSplit respects temporal order
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test
```

**test_context_dependent/manual_split.py** - Ambiguous case:
```python
import pandas as pd

def prepare_timeseries_data(df):
    """Manual 80/20 split - temporal order unclear"""
    split_idx = int(len(df) * 0.8)
    train = df[:split_idx]
    test = df[split_idx:]

    # Context-dependent: OK if df is already sorted chronologically,
    # problematic if user sorted by something else first
    return train, test
```

Now reference these tests in `pattern.toml`:

```toml
[tests]

[[tests.positive]]
file = "test_positive/shuffled_timeseries.py"
description = "Time-series data split with shuffle=True"
expected_issue = "Temporal leakage: shuffle=True on time-series data"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"
name = "prepare_timeseries_data"
snippet = "shuffle=True"

[[tests.negative]]
file = "test_negative/chronological_split.py"
description = "Correct time-series split using TimeSeriesSplit"
max_false_positives = 0
notes = "Uses sklearn's TimeSeriesSplit which maintains temporal order"

[[tests.context_dependent]]
file = "test_context_dependent/manual_split.py"
description = "Manual split - chronological order unclear from code"
allow_detection = true
allow_skip = true
context_notes = """
Detection is acceptable if the LLM can infer temporal data from context.
Not detecting is also OK since the code doesn't explicitly violate temporal ordering.
"""
```

**Test case guidelines:**
- **Positive**: Must be a clear, unambiguous bug
- **Negative**: Must be demonstrably correct code
- **Context-dependent**: Genuinely ambiguous or context-dependent cases only
- Include 2-5 positive cases, 1-3 negative cases minimum
- Add comments in test files explaining why they're buggy/correct

#### Step 5: Validate Your Pattern

**5.1 Validate TOML structure:**
```bash
python -m scicode_lint.tools.validate_pattern \
    patterns/ai-training/ml-050-temporal-split-leakage
```

Expected output:
```
ai-training/
────────────────────────────────────────────────────────────
  ✓ ml-050-temporal-split-leakage

VALIDATION SUMMARY
============================================================
Total patterns:    1
Valid:             1
Invalid:           0
```

**5.2 Rebuild the pattern registry:**
```bash
python -m scicode_lint.tools.rebuild_registry
```

This updates `patterns/_registry.toml` with your new pattern.

**5.3 Test with the linter (requires vLLM server):**
```bash
# Start vLLM server first
vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic

# Run linter on your test files
scicode-lint check patterns/ai-training/ml-050-temporal-split-leakage/positive/
scicode-lint check patterns/ai-training/ml-050-temporal-split-leakage/negative/
```

**Expected results:**
- Positive tests: Should trigger warnings
- Negative tests: Should have no warnings

**5.4 Run formal evaluation:**

The project has a **two-tier evaluation system**:

**Pattern-Specific Evaluation** (test your pattern in isolation):
```bash
# Hardcoded ground truth (fast, deterministic)
python evals/run_eval.py --pattern ml-050

# LLM-as-judge (semantic correctness)
python evals/run_eval_llm_judge.py --pattern ml-050
```

**Integration Evaluation** (test multi-pattern detection):
```bash
# Run all integration scenarios
python evals/integration/run_integration_eval.py -v

# Or with LLM judge
python evals/integration/run_integration_eval_llm_judge.py -v
```

This calculates precision, recall, and F1 score for your pattern.

**See also:**
- [evals/README.md](evals/README.md) - Pattern-specific evaluation details
- [evals/integration/README.md](evals/integration/README.md) - Integration evaluation details

#### Step 6: Quality Targets

Your pattern should meet these metrics:

| Metric | Target | Critical Severity |
|--------|--------|-------------------|
| **Precision** | ≥ 0.90 | ≥ 0.95 |
| **Recall** | ≥ 0.80 | ≥ 0.80 |
| **F1 Score** | ≥ 0.85 | ≥ 0.87 |

If metrics are below target:
- **Low precision**: Detection question is too broad, add more constraints
- **Low recall**: Detection question is too narrow, broaden scope or add more positive cases
- **Both low**: Rethink the detection approach

**💡 Pro tip:** Use the Pattern Reviewer agent to get improvement suggestions:
```bash
# Review your pattern comprehensively
claude --agent pattern-reviewer "Review ml-050-temporal-split-leakage"

# Or use the helper script
./scripts/review_patterns.sh review ml-050-temporal-split-leakage

# Get specific help with metrics
claude --agent pattern-reviewer "Precision for ml-050 is 0.85 (below 0.90). Suggest improvements."
```

See [docs_dev_genai/TOOLS.md](docs_dev_genai/TOOLS.md) for more about the Pattern Reviewer agent.

#### Step 7: Submit Your Pattern

1. **Commit your changes:**
```bash
git add patterns/ai-training/ml-050-temporal-split-leakage/
git add patterns/_registry.toml
git commit -m "Add pattern ml-050: temporal split leakage detection

- Detects shuffle=True on time-series data
- Precision: 0.95, Recall: 0.85, F1: 0.90
- 3 positive tests, 2 negative tests, 1 context-dependent test"
```

2. **Push and create PR:**
```bash
git push origin feature/ml-050-temporal-leakage
```

3. **In your PR description, include:**
   - Pattern ID and description
   - Why this pattern is important (research impact)
   - Evaluation metrics
   - Any edge cases or limitations

### Tips and Best Practices

#### Choosing Pattern IDs

Pattern IDs follow the format: `{category-prefix}-{number}`. Check `patterns/_registry.toml` for next available IDs.

#### Common Pitfalls to Avoid

❌ **Don't**: Write detection questions that require deep domain knowledge
```toml
question = "Will this cause overfitting in production?"
```

✅ **Do**: Focus on observable code patterns
```toml
question = "Is the validation set used during hyperparameter tuning?"
```

❌ **Don't**: Create overlapping patterns
- Check if a similar pattern already exists
- Combine closely related checks into one pattern

✅ **Do**: Make patterns orthogonal and focused
- Each pattern detects one specific bug type
- Use `related_patterns` to link similar issues

❌ **Don't**: Use placeholder test files
```python
# TODO: Add real test case
def placeholder():
    pass
```

✅ **Do**: Use realistic, representative code
```python
# Realistic example from scikit-learn documentation
from sklearn.preprocessing import StandardScaler
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X)  # BUG: fit on all X
```

#### Getting Help

If you're stuck or unsure:
1. Look at existing patterns for inspiration (especially `ml-001-scaler-leakage`)
2. Open a draft PR early for feedback
3. Ask in GitHub issues if you have questions about pattern design

### Writing Good Detection Questions

**Good detection questions**:
- Are specific and unambiguous
- Focus on observable code patterns
- Avoid requiring deep domain knowledge from the LLM
- Can be answered by reading the code in context

**Example - Good**:
```
Is there a call to StandardScaler.fit_transform() or .fit() that occurs before
train_test_split() in the code flow?
```

**Example - Bad**:
```
Is there data leakage in this machine learning pipeline?
```
(Too vague - requires LLM to understand what constitutes "data leakage")

### Writing Good Warning Messages

**Good warning messages**:
- Explain the problem clearly
- State why it matters (impact on results)
- Provide concrete fix suggestions
- Are readable by domain researchers

**Example**:
```
Data leakage: StandardScaler.fit_transform() called before train_test_split().
The scaler learns statistics from test data, inflating model performance.
Use sklearn.pipeline.Pipeline so fitting happens inside each fold.
```

## Running Tests

### Unit Tests

```bash
pytest tests/ -v
```

### Evaluation Framework

The project has a **two-tier evaluation system**:

**1. Pattern-Specific Evaluations** (`evals/`)
- Tests individual patterns in isolation
- Both hardcoded ground truth and LLM-as-judge approaches

```bash
# Hardcoded ground truth (fast)
python evals/run_eval.py --pattern ml-001

# LLM-as-judge (semantic)
python evals/run_eval_llm_judge.py --pattern ml-001

# All patterns
python evals/run_eval.py

# As pytest
pytest evals/test_all_patterns.py -v
```

**2. Integration Evaluations** (`evals/integration/`)
- Tests multi-pattern detection on realistic code
- Validates patterns work together without interference

```bash
# Hardcoded ground truth
python evals/integration/run_integration_eval.py -v

# LLM-as-judge
python evals/integration/run_integration_eval_llm_judge.py -v
```

**See:** [evals/README.md](evals/README.md) and [evals/integration/README.md](evals/integration/README.md) for details.

## Code Quality

Run type checking (`mypy src/`) and linting (`ruff check . && ruff format .`) before committing.

Ensure all checks pass and documentation is updated.

## Submitting Changes

1. Create a feature branch
2. Make your changes following the guidelines above
3. Commit with clear messages
4. Push to your fork
5. Open a Pull Request with:
   - Description of changes
   - Related issues
   - Test results for detection patterns

### PR Checklist

- [ ] Code follows project style (ruff, mypy pass)
- [ ] Tests added/updated and passing
- [ ] Documentation updated (README, docstrings, CHANGELOG)
- [ ] New detection patterns have test cases
- [ ] Pattern-specific evaluation metrics meet quality targets (≥0.90 precision, ≥0.80 recall)
- [ ] Integration tests pass (if modifying core linter code)

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be specific and include examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
