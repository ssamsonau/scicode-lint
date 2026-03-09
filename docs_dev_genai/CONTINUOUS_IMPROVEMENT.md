# Continuous Improvement Loop

**Key principle:** Don't fine-tune to specific tests. Extract generalizable principles and apply to ALL patterns.

**Requires:** vLLM server running (runs actual detection against test files)

**Different from pattern review:** Pattern review (`claude --agent pattern_verification/semantic/pattern-reviewer`) is static analysis of pattern files that does NOT run evals. Use it independently to check pattern structure, test file quality, etc.

## Loop

```bash
# 0. Run validation script first (catches structural issues before eval)
python pattern_verification/deterministic/validate.py
# Fix any errors before proceeding. The script checks:
# TOML/file sync, schema, data leakage, test count, TODO markers,
# detection format, syntax, empty fields, diversity

# 1. Evaluate individual patterns
python evals/run_eval.py

# 2. Debug FAILING patterns (see LLM reasoning)
python -m scicode_lint check <file> --pattern <id> --verbose

# 3. Fix - but look for GENERAL principle, not specific fix
#    If found generalizable principle:
#    - Add to patterns/README.md
#    - Update pattern-reviewer agent if needed
#    - Apply to ALL pattern questions, not just the failing one

# 4. (Optional) Review changed patterns for structural issues
#    ⚠️ COST: Uses Opus API. Prefer single-pattern reviews.
claude --agent pattern_verification/semantic/pattern-reviewer "Review <pattern-id>"

# 5. Validate - check nothing broke
python evals/run_eval.py

# 6. If updated pattern-reviewer agent, run on all patterns to verify
#    ⚠️ COST: Bulk reviews consume significant tokens. Use sparingly.
claude --agent pattern_verification/semantic/pattern-reviewer "Review all patterns"
```

## Parallel Category Loops

Run improvement loops per category in parallel (each in separate terminal):

```bash
# Terminal 1: AI inference patterns
python evals/run_eval.py -c ai-inference

# Terminal 2: AI training patterns
python evals/run_eval.py -c ai-training

# Terminal 3: Scientific numerical patterns
python evals/run_eval.py -c scientific-numerical

# Terminal 4: Scientific performance patterns
python evals/run_eval.py -c scientific-performance

# Terminal 5: Scientific reproducibility patterns
python evals/run_eval.py -c scientific-reproducibility
```

Each category writes reports to `evals/reports/judge/<category>/`.

**Note:** All parallel runs share the same vLLM server. Start the server first with sufficient capacity, or let the first process auto-start it.

## Integration Tests (holdout set)

Tests generalization - did we overfit to pattern-specific tests?

```bash
python evals/integration/run_integration_eval.py     # static integration
python evals/integration/dynamic_eval.py             # dynamic integration
```

## Update README

When metrics change, update README.md.

## Evaluation Types

| Type | Purpose |
|------|---------|
| Pattern-specific | Iterate on individual patterns |
| Static integration | Holdout - test all patterns on realistic code |
| Dynamic integration | Holdout - test on fresh LLM-generated code |

## Critical Constraint

- **No hints in test files** - pure code only, no comments about bugs (data leakage in evaluation)
