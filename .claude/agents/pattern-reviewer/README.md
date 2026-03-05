# Pattern Reviewer Agent

A specialized Claude agent for reviewing, improving, and creating test cases for scicode-lint pattern definitions.

**⚠️ Requirements:** This agent requires [Claude Code CLI](https://github.com/anthropics/claude-code) to function. It is specifically designed as a Claude Code agent and will not work with other tools or interfaces.

## Purpose

This agent helps maintain high quality pattern definitions by:
- Reviewing pattern TOML files for completeness and clarity
- Analyzing test cases (positive, negative, context-dependent)
- Suggesting improvements to detection questions and warning messages
- Creating new test cases to improve coverage
- Identifying missing edge cases and false positive risks
- Validating pattern structure and metadata

## Usage

### Option 1: Using Claude Code CLI

```bash
# Review a specific pattern
claude --agent pattern-reviewer "Review the ml-001-scaler-leakage pattern"

# Review all patterns in a category
claude --agent pattern-reviewer "Review all patterns in ai-training category"

# Create new test cases
claude --agent pattern-reviewer "Create additional negative test cases for ml-001"

# General pattern improvement
claude --agent pattern-reviewer "Find patterns that need better detection questions"
```

### Option 2: Using in Claude Code Interactive Session

```
/agent pattern-reviewer

What would you like me to review?
1. Specific pattern (e.g., ml-001-scaler-leakage)
2. All patterns in a category (e.g., ai-training)
3. Find patterns needing improvement
4. Create new test cases for a pattern
```

## What This Agent Does

### 1. Pattern Definition Review

Checks each `pattern.toml` for:
- ✅ Complete metadata (id, name, category, severity, version)
- ✅ Clear and focused detection question
- ✅ Actionable warning message
- ✅ Appropriate severity level
- ✅ Comprehensive explanation
- ✅ Relevant tags and related patterns
- ✅ Quality targets (precision, recall)

### 2. Test Case Analysis

⚠️ **CRITICAL**: Test files must contain ONLY executable code - NO docstrings, NO comments explaining bugs or fixes. All context belongs in `pattern.toml` ([detection] and [tests] sections).

⚠️ **ANTI-PATTERN**: Test files should NOT use the exact syntax examples from the detection question. This creates evaluation data leakage where the LLM is told "look for X" and the test literally contains X.

**Example of what to avoid:**
- ❌ Detection question: "Look for `torch.cat([train, test])`"
- ❌ Test file: `all_data = torch.cat([train_data, test_data])`
- ✅ Test file: `full_dataset = torch.vstack([train_data, test_data])` (same bug, different syntax)

Reviews test files in:
- **test_positive/** - Code with issues (must detect)
  - Checks for realistic scenarios
  - Ensures good coverage of variations
  - Verifies pure code with no hints
  - **NEW:** Verifies no exact syntax match with detection question examples

- **test_negative/** - Correct code (must NOT detect)
  - Checks for comprehensive edge cases
  - Ensures similarity to positive cases
  - Verifies pure code with no hints
  - **NEW:** Verifies varied implementations of correct patterns

- **test_context_dependent/** - Ambiguous cases
  - Validates nuanced scenarios
  - Checks for clear context notes in pattern.toml [tests] section

### 3. Improvement Suggestions

Provides:
- Better detection questions (more focused, less ambiguous)
- Clearer warning messages (more actionable)
- New test case ideas (missing scenarios)
- Severity adjustments (if mismatched)
- Related pattern suggestions
- Tag improvements

### 4. New Test Case Creation

Creates:
- Realistic code examples (pure Python, NO docstrings)
- Multiple variations of the same issue
- Edge cases and boundary conditions
- False positive prevention cases
- Context descriptions in `pattern.toml` [tests] section (NOT in test files)

## Example Workflows

### Review a Single Pattern

```bash
# Review pattern ml-001
claude --agent pattern-reviewer "Review ml-001-scaler-leakage comprehensively"
```

Output includes:
- Pattern summary and status
- Strengths of current definition
- Issues found with specific suggestions
- Missing test cases
- New test case code examples

### Review All Patterns in a Category

```bash
# Review all ai-training patterns
claude --agent pattern-reviewer "Review all patterns in ai-training category and create a summary report"
```

Output includes:
- Category overview
- Pattern-by-pattern analysis
- Common issues across patterns
- Recommended improvements
- Priority list for fixes

### Batch Review Multiple Patterns

```bash
# Review specific patterns in parallel
claude --agent pattern-reviewer "Review ml-001, ml-002, and ml-003 in parallel"

# Review all critical severity patterns
claude --agent pattern-reviewer "Review all critical severity patterns and identify top 3 priorities"

# Find patterns matching criteria
claude --agent pattern-reviewer "Find and review all patterns with fewer than 3 test cases"
```

**Parallel processing:** The agent can analyze multiple patterns concurrently and provide:
- Individual summaries for each
- Common issues across all reviewed patterns
- Prioritized action list
- Category or severity-level insights

See [BATCH_OPERATIONS.md](BATCH_OPERATIONS.md) for detailed guide on batch processing.

### Create New Test Cases

```bash
# Add edge cases for a pattern
claude --agent pattern-reviewer "Create 3 new negative test cases for ml-001 that test common false positive scenarios"
```

Output includes:
- New Python files with pure code (NO docstrings)
- Context descriptions for pattern.toml [tests] section
- Suggestions for where to add them

### Find Patterns Needing Work

```bash
# Identify improvement opportunities
claude --agent pattern-reviewer "Which patterns have incomplete test coverage or unclear detection questions?"
```

Output includes:
- List of patterns with issues
- Severity of issues
- Recommended actions
- Priority order

## Directory Structure

```
.claude/agents/pattern-reviewer/
├── agent.json              # Agent configuration
├── system_prompt.md        # Agent instructions and expertise
└── README.md              # This file
```

## Pattern Quality Standards

The agent enforces these quality targets:

| Metric | Target | Critical Severity |
|--------|--------|-------------------|
| Precision | ≥ 0.90 | ≥ 0.95 |
| Recall | ≥ 0.80 | ≥ 0.80 |
| F1 Score | ≥ 0.85 | ≥ 0.87 |

## Common Issues the Agent Finds

1. **Vague detection questions**
   - Too broad or multi-part
   - Not focused enough for LLM evaluation

2. **Non-actionable warnings**
   - Describes problem but not solution
   - Missing specific fix recommendations

3. **Missing test cases**
   - No negative cases for obvious fixes
   - No context-dependent cases for nuanced issues
   - Missing variations of the same bug

4. **Severity mismatches**
   - Critical issues marked as high/medium
   - Medium issues marked as critical

5. **Incomplete metadata**
   - Missing tags or related patterns
   - No references or documentation
   - Incomplete quality targets

6. **⚠️ Evaluation data leakage (NEW)**
   - Test files use exact syntax from detection question examples
   - Variable names match detection question patterns too closely
   - Example: Question says "look for `torch.cat([train, test])`" and test has `torch.cat([train_data, test_data])`
   - **Fix:** Use equivalent operations with different syntax (e.g., `torch.vstack`, manual array operations)

## Integration with Evaluation Framework

The agent works alongside `evals/run_eval.py`:

```bash
# 1. Review and improve patterns
claude --agent pattern-reviewer "Review ml-001"

# 2. Run evaluation to measure improvements
python evals/run_eval.py --pattern ml-001-scaler-leakage

# 3. Iterate based on results
claude --agent pattern-reviewer "The precision for ml-001 is 0.87, below target. Suggest improvements."
```

## Best Practices for Pattern Quality

### Avoiding Evaluation Data Leakage

When detection questions provide technical examples (e.g., "look for `torch.cat([train, test])`"), test files should use **functionally equivalent but syntactically different** implementations:

| Detection Question Examples | Test File Should Use |
|----------------------------|----------------------|
| `torch.cat([train, test])` | `torch.vstack()`, `torch.stack()` |
| `np.concatenate([train, test])` | `np.vstack()`, `np.hstack()` |
| `df.groupby('cat')['target'].mean()` | `df.groupby('cat').agg({'target': 'mean'})` |
| `shift(-1)` | Manual array slicing: `.values[1:]` |
| `scaler.fit_transform(X_test)` | Different scaler names: `normalizer`, `preprocessor` |

**Why this matters:** If test files contain the exact syntax from examples, the LLM sees the answer directly, invalidating the evaluation.

## Tips for Best Results

1. **Be specific** in your requests
   - Good: "Review ml-001 detection question for clarity"
   - Less good: "Review some patterns"

2. **Provide context** when creating test cases
   - Good: "Create negative cases that test train/val/test split edge cases"
   - Less good: "Create negative cases"

3. **Check for syntax overlap** between detection questions and test files
   - Good: "Check if ml-001 test files use different syntax than detection question examples"
   - This prevents evaluation data leakage

4. **Iterate** based on evaluation results
   - Review → Improve → Evaluate → Repeat

5. **Focus on research context**
   - Patterns should reflect real scientific code issues
   - Not general Python antipatterns

## Contributing

To improve this agent:

1. Update `system_prompt.md` for better expertise
2. Add examples to this README
3. Extend `agent.json` with new capabilities
4. Test with various pattern review scenarios

## See Also

- [patterns/README.md](../../../patterns/README.md) - Pattern structure documentation
- [evals/](../../../evals/) - Evaluation framework
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Contribution guidelines
