# scicode-lint

AI-powered linter for scientific Python code using local LLM (Qwen3 via vLLM).

> **Keep this file focused.** Rules and critical info only. No directory structures, no redundant examples. Claude can explore the codebase itself.

---

## Key Files

**Main files (root):**
- [README.md](./README.md) - Main file for humans
- [INSTALLATION.md](./INSTALLATION.md) - Detailed installation guide
- [DOCUMENTATION_MAP.md](./DOCUMENTATION_MAP.md) - Navigation guide to all documentation
- [CLAUDE.md](./CLAUDE.md) - This file, main instructions for AI agents working on the codebase
- [patterns/](./patterns/) - Detection patterns (64 patterns across 5 categories)

**For humans (docs_use_human/):**
- [README.md](docs_use_human/README.md) - Documentation index
- [USAGE.md](docs_use_human/USAGE.md) - CLI usage guide
- [VRAM_REQUIREMENTS.md](docs_use_human/VRAM_REQUIREMENTS.md) - Hardware requirements
- [performance/](docs_use_human/performance/) - Performance & benchmarking guides

**For AI agents USING scicode-lint (docs_use_genai/):**
- [GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md) - Complete guide for AI agents using scicode-lint
- [INTERFACE_ANALYSIS.md](docs_use_genai/INTERFACE_ANALYSIS.md) - Package interface analysis

**For AI agents WORKING ON scicode-lint (docs_dev_genai/):**
- [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) - Core design principles (READ THIS FIRST)
- [CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md) - Evaluation → improvement workflow
- [IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md) - Technical implementation details

**Pattern verification (pattern_verification/):**
- [pattern_verification/](pattern_verification/) - Deterministic + semantic pattern quality checks

---

## Architecture Overview

**Core principle: Constrained-capacity local LLM by design.**

scicode-lint uses a small local model (fits in 16GB VRAM) - a deliberate choice between grep-style matching and expensive SOTA models. This enables local execution, privacy, no API costs, and reproducibility (open-source models remain available), but requires detection questions to be simple and direct.

**Two-phase design:**
- **Build time:** Design detection questions that constrained models can reliably answer
- **Runtime:** Local model (FP8 via vLLM) runs focused checks against code

**Philosophy:** Detection only, no automatic fixes.

**⚠️ CRITICAL ARCHITECTURAL RULES:**
- Code MUST come before detection instructions in prompts (enables vLLM prefix caching)
- Detection questions must be focused (one issue per question) with self-contained context
- Include "why it matters" directly in the question (Qwen3 is a thinking model - context helps reasoning)

**📖 Complete architecture:** [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)

---

## Local LLM Setup

**Required:** vLLM server with FP8 model (16GB+ VRAM, native FP8 support required)

```bash
pip install scicode-lint[vllm-server]
bash src/scicode_lint/vllm/start_vllm.sh   # Auto-detects GPU, validates FP8 support (default port: 5001)
python -m scicode_lint check myfile.py
```

**📖 Complete setup:** [INSTALLATION.md](INSTALLATION.md)

---

## Pattern Development

**📖 Pattern guide:** [patterns/README.md](patterns/README.md) - Structure, detection question template, test file rules

### Pattern Verification

**📖 Full guide:** [pattern_verification/README.md](pattern_verification/README.md)

Two-step verification (does NOT run evals):

```bash
# 1. Deterministic checks (fast, no LLM)
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/validate.py --fix  # auto-fix

# 2. Semantic review (uses LLM for consistency checking)
claude --agent pattern-reviewer "Review ml-001-scaler-leakage"
```

**Bulk review (all patterns):** When asked to "review all patterns", launch one agent per pattern in parallel (64 concurrent agents). Run deterministic checks first.

### Improvement Loop (runs evals, requires vLLM)

Evaluates detection accuracy and iteratively improves patterns. See [Continuous Improvement Loop](#continuous-improvement-loop) below.

---

## Critical Development Rules

### Code Quality
- **DRY (Don't Repeat Yourself)** - Define constants, configs, and model names once; reference them everywhere
- **YAGNI (You Aren't Gonna Need It)** - Don't add features "just in case"; build for current requirements only
- **Fail fast** - Raise errors early with clear messages; don't let invalid state propagate
- **No silent fallbacks** - If config cannot be loaded, FAIL. Never silently use default values when config loading fails
- **Explicit over implicit** - Make intent clear; avoid magic numbers, use named constants
- **Small functions** - Each function does one thing; if a block needs a comment, extract it to a named function
- **Reasonable file size** - Split files > 1000 lines; each file has one clear responsibility
- **Meaningful names** - Names describe purpose, not type (`max_retries` not `n`, `user_ids` not `list1`)
- **NEVER hide issues** - Investigate root cause, don't suppress without understanding
- **Known issues require references** - Cite specific GitHub issue/docs (verify it exists, no made-up citations)
- **NO LEGACY CODE** - Never create "legacy/", "old/", "deprecated/" directories - delete unused code completely
- **Dual audience** - Detection output must be clear for BOTH humans and GenAI agents
- Every detection pattern must have eval coverage

### Prompt Files & Linting
- **NEVER modify prompt content to fix lint errors** - Prompt wording affects model behavior
- If ruff reports E501 (line too long) in a prompt file, add per-file-ignores in pyproject.toml:
  ```toml
  [tool.ruff.lint.per-file-ignores]
  "path/to/prompts.py" = ["E501"]
  ```

### Git Commits
- **Do NOT include Claude as co-author** in commit messages (no `Co-Authored-By: Claude` lines)

### Initial Setup (once per clone)
```bash
git config core.hooksPath scripts  # Enable pre-commit hooks (ruff + mypy)
```

### Mandatory Checks (RUN AFTER EVERY CODE CHANGE)
```bash
ruff check . && ruff format .  # Code style
mypy .                         # Type checking (patterns/ excluded via pyproject.toml)
pytest                         # Tests
```

### Type Hints & Documentation
- **All functions** must have type annotations (args + return)
- **All exported classes/methods** (in `__all__`) must have docstrings (Args, Returns, Examples)
- **Update API_REFERENCE.md** when adding public APIs
- Private methods (_prefixed) need type hints only

### Documentation Rules
- **Docs are snapshots** - Describe current state, not history (history lives in git)
- **No "improved from" language** - Just state what IS, not what changed (e.g., "Precision: 34%" not "Precision: 34% (improved from 24%)")
- **patterns/README.md** - Pattern guide with detection question template
- **Check after every change** - Fix documentation drift immediately
- **Human docs:** Concise, link to external docs, scicode-lint-specific only
- **GenAI docs:** Comprehensive, self-contained, complete examples, explain rationale

### Project Principles
- **NO backward compatibility** - Clean breaks over cruft (young project in active development)
- **Minimize false positives** - A noisy tool gets ignored
- **Keep it simple** - Modern Python packaging (pyproject.toml, ruff, pytest)

**📖 Complete guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)

**License:** MIT

---

## AI Coding Agent Workflow

After writing code, **always** run:

1. **Document** - Add docstrings + type hints to public APIs, update API_REFERENCE.md if in `__all__`
2. **Quality** - `ruff check . && ruff format .` then `mypy .`
3. **Test** - `pytest`
4. **Docs** - Update docs to match current state (no "updated"/"changed" language)
5. **Commit** - Only commit clean, type-safe, documented code

---

## Continuous Improvement Loop

**Also called:** improvement loop, eval loop, pattern improvement, improve patterns, run evals, do evals

**What it is:** A structured workflow for iteratively evaluating and improving detection pattern quality. Requires vLLM server running.

**Different from pattern review:** Pattern review (above) is static analysis of pattern files. Improvement loop runs actual detection against test files.

**📖 Complete instructions:** [docs_dev_genai/CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md)

**Quick start:**
```bash
# 1. Evaluate patterns
python evals/run_eval.py

# 2. Debug failing patterns
python -m scicode_lint check <file> --pattern <id> --verbose

# 3. Fix and validate
python evals/run_eval.py

# 4. Integration tests (holdout - test generalization)
python evals/integration/run_integration_eval.py
python evals/integration/dynamic_eval.py
```
