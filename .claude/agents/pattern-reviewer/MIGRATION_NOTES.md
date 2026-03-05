# Pattern Reviewer Agent - Migration to New Structure

**Date**: 2026-03-04

## Changes Made

This document summarizes the updates to the pattern-reviewer agent to reflect the new pattern structure and NO DATA LEAKAGE policy.

## Directory Structure Changes

### Old Structure
```
patterns/{category}/{pattern-id}/
├── pattern.toml
├── positive/              # Code with bugs
├── negative/              # Correct code
└── context_dependent/     # Edge cases
```

### New Structure
```
patterns/{category}/{pattern-id}/
├── pattern.toml                 # Complete pattern definition
│                                # [meta] + [detection] → linter sees
│                                # [tests] → evaluator sees
├── test_positive/               # Code WITH issues
├── test_negative/               # Code WITHOUT issues
└── test_context_dependent/      # Edge cases
```

**Note**: No separate evaluation.yaml needed. All test definitions go in pattern.toml [tests] section.

## Critical Policy: NO DATA LEAKAGE

### The Problem
If test files contain documentation like:
```python
"""
BUG: This causes data leakage
"""
def problematic_function():
    # Data leakage here
    ...
```

The LLM sees the answer in the code itself, invalidating evaluation.

### The Solution
Test files now contain **ONLY executable Python code**:
```python
def problematic_function():
    ...
```

All context goes in **pattern.toml sections**:
- `pattern.toml` [detection] - What to look for (linter sees this)
- `pattern.toml` [tests] - Expected behavior (evaluator sees this, linter does NOT)

## What Changed in Agent Files

### system_prompt.md
- ✅ Updated directory structure diagram
- ✅ Added strong NO DATA LEAKAGE warning
- ✅ Updated all file path examples: `positive/` → `test_positive/`, etc.
- ✅ Removed references to "comprehensive docstrings" in test files
- ✅ Added section on detection question quality (ANALYSIS APPROACH)
- ✅ Updated quality standards to emphasize pure code only

### examples.md
- ✅ Removed ALL docstrings from example test code
- ✅ Added notes showing context belongs in pattern.toml [tests] section
- ✅ Updated all file paths to new structure
- ✅ Converted code examples to pure Python
- ✅ Changed YAML examples to TOML format

### README.md
- ✅ Updated test case analysis section with NO DATA LEAKAGE warning
- ✅ Changed directory names throughout
- ✅ Removed references to docstrings in test files
- ✅ Updated "See Also" links: specs/ → patterns/

### QUICK_START.md
- ✅ Updated "See Also" links: specs/ → patterns/

## Key Agent Behaviors

### When Reviewing Patterns
The agent now checks that:
- Test files contain ONLY code (no docstrings, no comments)
- Context is in pattern.toml [detection] and [tests] sections
- Detection questions encourage understanding code structure first
- File paths use test_positive/, test_negative/, test_context_dependent/
- No separate evaluation.yaml file needed

### When Creating Test Cases
The agent now:
- Creates pure Python code with NO documentation
- Puts all context in pattern.toml [tests] section
- Uses correct directory names (test_*)
- Warns if ANY hints appear in test files
- Does NOT create separate evaluation.yaml files

### When Suggesting Improvements
The agent now:
- Suggests improving detection questions to avoid false positives
- Recommends adding ANALYSIS APPROACH guidance
- Never suggests adding docstrings to test files
- Focuses on pattern.toml [detection] and [tests] sections for context

## Detection Quality Improvements

The new approach includes guidance for LLMs to:
1. First understand overall code structure and purpose
2. Identify intent of key operations
3. Trace data flow relevant to the question
4. THEN answer the specific detection question

This helps avoid false positives by ensuring context understanding before judgment.

## Files Updated
- `.claude/agents/pattern-reviewer/system_prompt.md`
- `.claude/agents/pattern-reviewer/examples.md`
- `.claude/agents/pattern-reviewer/README.md`
- `.claude/agents/pattern-reviewer/QUICK_START.md`

## Testing the Agent

After these changes, the agent should:
1. Only create test files with pure code
2. Flag any test files with docstrings as data leakage
3. Suggest improvements to detection questions
4. Use correct directory structure in all recommendations

## Key Takeaway

**Single File Approach**: Everything in pattern.toml with logical separation:
- Linter reads: `[meta]` + `[detection]` sections
- Evaluator reads: `[tests]` section
- No separate evaluation.yaml needed

This keeps patterns self-contained while maintaining separation of concerns.

## See Also
- [patterns/README.md](../../../patterns/README.md) - Pattern structure documentation
- [src/scicode_lint/detectors/prompts.py](../../../src/scicode_lint/detectors/prompts.py) - Detection prompts with ANALYSIS APPROACH
