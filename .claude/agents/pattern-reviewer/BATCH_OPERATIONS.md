# Batch Operations Guide

The pattern-reviewer agent supports efficient batch processing of multiple patterns concurrently.

**⚠️ Prerequisites:** Requires [Claude Code CLI](https://github.com/anthropics/claude-code)

---

## Overview

Instead of reviewing patterns one-by-one sequentially, you can review multiple patterns in a single request. The agent will:

1. **Process in parallel** - Analyze multiple patterns concurrently
2. **Aggregate insights** - Find common issues across patterns
3. **Prioritize actions** - Rank patterns by urgency/impact
4. **Provide batch summary** - Category or severity-level insights

---

## Batch Review Methods

### 1. Specific Pattern List

Review an explicit list of patterns:

```bash
claude --agent pattern-reviewer "Review ml-001, ml-002, ml-003, and ml-007 in parallel"
```

**Use when:**
- You have a specific set of related patterns
- Working on a feature that touches multiple patterns
- Following up on evaluation results

### 2. By Category

Review all patterns in a category:

```bash
# All AI training patterns
claude --agent pattern-reviewer "Review all patterns in ai-training category"

# All numerical patterns
claude --agent pattern-reviewer "Review all patterns in scientific-numerical category"
```

**Available categories:**
- `ai-training` (15 patterns)
- `ai-inference` (3 patterns)
- `ai-data` (1 pattern)
- `scientific-numerical` (10 patterns)
- `scientific-reproducibility` (4 patterns)
- `scientific-performance` (11 patterns)

### 3. By Severity

Review all patterns of a specific severity:

```bash
# All critical severity patterns
claude --agent pattern-reviewer "Review all critical severity patterns and prioritize improvements"

# All high severity patterns
claude --agent pattern-reviewer "Review all high severity patterns"
```

**Severity levels:**
- `critical` - Data correctness, scientific validity issues
- `high` - Reproducibility, significant bugs
- `medium` - Performance, code quality

### 4. Find Patterns Matching Criteria

Search and review patterns matching specific criteria:

```bash
# Patterns with incomplete test coverage
claude --agent pattern-reviewer "Find and review all patterns with fewer than 3 test cases"

# Patterns with low precision
claude --agent pattern-reviewer "Review all patterns where evaluation precision is below 0.90"

# Patterns missing metadata
claude --agent pattern-reviewer "Find patterns with incomplete TOML metadata (missing tags, references, or related patterns)"
```

---

## Batch Output Format

### Summary Section

```markdown
## Batch Review: 8 ai-training patterns

### Summary Statistics
- Total patterns reviewed: 8
- Status breakdown:
  - ✅ Good: 5
  - ⚠️ Needs improvement: 2
  - ❌ Issues found: 1

### Priority List (Top 5)
1. ml-001-scaler-leakage - CRITICAL: Truncated description, only 1 test case
2. ml-003-cross-validation - HIGH: Detection question too broad
3. ml-007-temporal-leakage - MEDIUM: Missing negative test cases
4. ml-005-metric-mismatch - LOW: Missing references
5. ml-008-class-imbalance - LOW: Tags could be improved
```

### Common Issues

```markdown
### Common Issues Across Patterns

#### Missing Test Cases (3 patterns)
- ml-001, ml-007, ml-009: Need more negative test cases
- Recommended: Add 2-3 negative cases per pattern showing correct approaches

#### Vague Detection Questions (2 patterns)
- ml-003, ml-004: Questions ask multiple things at once
- Recommended: Split into focused yes/no questions

#### Incomplete Metadata (4 patterns)
- ml-005, ml-006, ml-008, ml-009: Missing references or tags
- Recommended: Add relevant references to documentation
```

### Individual Pattern Reviews

```markdown
### Individual Pattern Reviews

#### ml-001-scaler-leakage
**Status:** ❌ Issues Found
**Priority:** 1 (High Impact)

**Issues:**
1. Truncated description field
2. Only 1 positive test case (need 3-4)
3. No negative test cases

**Recommended Actions:**
[Detailed actions]

---

#### ml-002-feature-selection
**Status:** ✅ Good

**Strengths:**
- Complete TOML metadata
- 4 positive test cases covering variations
- 3 negative test cases
- Clear detection question

**Minor Suggestions:**
- Add context-dependent case for feature selection on train+val

---

[Continue for all patterns...]
```

---

## Usage Examples

### Example 1: Review Critical Patterns Before Release

```bash
claude --agent pattern-reviewer "Review all critical severity patterns. For each pattern, verify: (1) detection question is clear and focused, (2) at least 3 test cases of each type, (3) evaluation precision ≥ 0.95. Prioritize any failing these criteria."
```

**Result:**
- Batch analysis of all critical patterns
- Pass/fail on each criterion
- Priority-ranked action list
- Specific fixes needed for failing patterns

### Example 2: Category Cleanup

```bash
claude --agent pattern-reviewer "Review all ai-training patterns. Identify common issues and suggest standardizations for: (1) detection question format, (2) warning message style, (3) tag conventions."
```

**Result:**
- Category-wide consistency analysis
- Common patterns in good vs problematic definitions
- Standardization recommendations
- Templates for consistent improvements

### Example 3: Post-Evaluation Improvement

```bash
claude --agent pattern-reviewer "Review patterns ml-001, ml-003, ml-005, and ml-007. These all have precision < 0.90 in latest evaluation. Analyze likely causes of false positives and suggest specific improvements to detection questions."
```

**Result:**
- Focused analysis on precision issues
- Pattern-by-pattern false positive analysis
- Specific detection question improvements
- Expected precision impact of changes

### Example 4: New Contributor Onboarding

```bash
claude --agent pattern-reviewer "Review patterns ml-001, ml-002, ml-003 as examples. For each, explain: (1) what makes the pattern well-designed, (2) how test cases are structured, (3) best practices demonstrated. Use these as templates for new patterns."
```

**Result:**
- Annotated examples of good patterns
- Best practice explanations
- Template guidance for new contributors
- Common pitfalls to avoid

---

## Performance Tips

### Optimal Batch Sizes

**Small batch (3-5 patterns):**
- Most detailed analysis per pattern
- Best for focused work on related patterns
- Typical completion: 2-3 minutes

**Medium batch (6-10 patterns):**
- Balanced detail and coverage
- Good for category reviews
- Typical completion: 4-6 minutes

**Large batch (11+ patterns):**
- Higher-level insights
- Best for finding common issues
- More summary, less detail per pattern
- Typical completion: 6-10 minutes

### When to Use Batch vs Sequential

**Use batch when:**
- ✅ Looking for common issues across patterns
- ✅ Prioritizing work across multiple patterns
- ✅ Reviewing a logical group (category, severity)
- ✅ Initial assessment of pattern quality

**Use sequential when:**
- ✅ Deep dive on a single complex pattern
- ✅ Creating comprehensive test cases
- ✅ Iterating on improvements after evaluation
- ✅ Detailed test case code generation

---

## Integration with Workflow

### Batch → Sequential Refinement

```bash
# 1. Batch review to find issues
claude --agent pattern-reviewer "Review all ai-training patterns and identify top 3 priorities"

# 2. Sequential deep dive on priorities
claude --agent pattern-reviewer "Review ml-001-scaler-leakage comprehensively and create missing test cases"

# 3. Iterate based on evaluation
python evals/run_eval.py --pattern ml-001-scaler-leakage

# 4. Sequential improvement
claude --agent pattern-reviewer "Precision for ml-001 is 0.87. Suggest improvements."
```

### Weekly Quality Check

```bash
# Monday: Batch review all categories
./scripts/review_patterns.sh review-category ai-training
./scripts/review_patterns.sh review-category scientific-numerical
# [etc for each category]

# Tuesday-Friday: Sequential fixes on priorities
# [Based on Monday's priority list]

# Friday: Batch evaluation
python evals/run_eval.py
```

---

## Batch Output vs Sequential Output

### Batch Review Output

**Strengths:**
- Overview of multiple patterns
- Common issues identified
- Relative priority ranking
- Category/severity insights

**Limitations:**
- Less detail per pattern
- Fewer specific code examples
- Higher-level suggestions

**Best for:**
- Initial assessment
- Finding patterns needing work
- Understanding category-wide issues
- Prioritizing work

### Sequential Review Output

**Strengths:**
- Deep analysis of single pattern
- Specific code examples
- Detailed improvement suggestions
- Complete test case generation

**Limitations:**
- No cross-pattern insights
- No relative prioritization
- Slower for multiple patterns

**Best for:**
- Implementing improvements
- Creating test cases
- Iterative refinement
- Post-evaluation fixes

---

## Helper Script Support

The `review_patterns.sh` script supports some batch operations:

```bash
# Review entire category (batch operation)
./scripts/review_patterns.sh review-category ai-training

# Find issues across all patterns (batch operation)
./scripts/review_patterns.sh find-issues
```

For more complex batch operations, use Claude Code CLI directly:

```bash
claude --agent pattern-reviewer "your custom batch request"
```

---

## See Also

- [README.md](README.md) - Complete agent documentation
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [examples.md](examples.md) - Detailed examples
- [../../../patterns/README.md](../../../patterns/README.md) - Pattern specifications
