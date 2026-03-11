# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-03-10

### Added
- **Semantic validation script** (`semantic_validate.py`) - batch pattern review using Claude Code CLI
- **Documentation grounding** - new `references` field in pattern.toml linking to official docs, with URL fetching/caching in deterministic validator
- **Doc caching** with two-stage cleaning pipeline for fetched reference documentation
- Extended test coverage

### Changed
- **Architecture principle**: Patterns must be grounded in official documentation
- **Improvement loop restructured**: Per-pattern fix cycle, `--fetch-refs` and `--clean-cache` options
- **Semantic reviewer**: Added documentation alignment check
- **Concurrency control**: Added semaphore to linter (150 limit)
- **Eval runner refactored**: Direct async linter API instead of subprocess calls

### Fixed
- Memory explosion in semantic validation from sub-agent spawning

## [0.1.4] - 2026-03-09

### Changed
- **Performance optimization**: max_completion_tokens to 4096, concurrency to 150
- **GPU memory utilization**: reduced to 0.85 for stability
- **Faster execution**: single pattern ~35s (was ~50s), full scan ~75s (was ~90s)

### Added
- Development Approach section in README (GenAI-native development methodology)
- Speed benchmark documentation with vLLM dashboard screenshot

## [0.1.3] - 2026-03-08

### Added
- **Deterministic pattern validation** (`pattern_verification/deterministic/validate.py`) - 9 structural checks for pattern quality
- **Dependency Health Checker** (`tools/check_dependencies.py`) - security auditing with pip-audit, safety, bandit, and deprecation warning capture
- **vLLM Monitoring Dashboard** (`tools/vllm_dashboard.py`) - Streamlit real-time metrics (requests, throughput, KV cache, GPU utilization) with time-series charts
- **Alignment Metrics** in evaluation - tracks agreement between direct metrics and LLM judge, highlights divergent cases
- **Dynamic integration evaluation** (`evals/integration/dynamic_eval.py`) - generates fresh test code via Claude to avoid overfitting
- **Improvement Insights notepad** (`docs_dev_genai/IMPROVEMENT_INSIGHTS.md`) - persistent learnings across sessions
- **Release script** (`scripts/release.sh`) - automates package build and GitHub release
- LLM reasoning output in verbose mode (`--verbose`)
- Three-tier evaluation methodology docs (pattern-specific → static integration → dynamic integration)

### Changed
- **Expanded pattern library to 64 patterns** (from 44 in v0.1.0)
- **Increased minimum test samples** to 3 positive and 3 negative per pattern for better eval coverage
- **Switched to thinking model** (Gemma 3 27B) for improved detection accuracy
- Consolidated evaluation scripts: merged `run_eval_llm_judge.py` into `run_eval.py` (use `--skip-judge` for fast eval)
- Enhanced improvement loop with insights step and debug-with-reasoning step
- Updated CONTRIBUTING.md with dependency groups and security checks

### Dependencies
- Added to `[dev]`: pip-audit, bandit, safety
- Added to `[dashboard]`: nvidia-ml-py, altair

### Fixed
- **pt-007-inference-without-eval**: counting approach for reliable detection (R: 0→1.0)
- **ml-007-test-set-preprocessing**: explicit test data variable identification
- **np-002-broadcasting-shape-mismatch**: semantic dimension mismatch focus (P: 0.33→1.0)

## [0.1.2] - 2026-03-06

### Added
- Continuous Improvement Loop documentation (`docs_dev_genai/CONTINUOUS_IMPROVEMENT.md`)
  - Structured workflow: evaluate → identify failures → analyze with agent → validate → re-evaluate
  - Clear criteria for pattern quality (precision ≥ 0.90, recall ≥ 0.80)
  - Mandatory rule: every pattern change requires running pattern-reviewer agent
  - Documentation for improving the agent itself when needed
  - Critical rules that must never be broken (data leakage detection, etc.)

### Changed
- Updated `CLAUDE.md` with brief reference to continuous improvement workflow
- Updated `docs_dev_genai/TOOLS.md` with link to new documentation

### Fixed
- **rep-001-incomplete-random-seeds**: Added proper test case for `train_test_split()` missing `random_state` (previous test cases were checking unrelated incomplete seeding issue)
- **pt-001-missing-train-mode**: Simplified detection question to improve recall (P=0→1.0, R=0→0.5)
- **pt-007-inference-without-eval**: Focused detection question on missing `model.eval()` in inference code
- **ml-007-test-set-preprocessing**: Made test data variable identification more explicit in detection question

## [0.1.1] - 2026-03-06

### Added
- New pattern: `pt-011-unscaled-gradient-accumulation` - detects gradient accumulation without proper loss scaling
- New pattern: `perf-002-array-allocation-in-loop` - detects inefficient array allocation inside loops
- New pattern: `perf-005-unnecessary-array-copy` - detects redundant array copies
- Project statistics generation script (`scripts/project_stats_generate.py`)
- Mandatory data leakage check in pattern reviewer agent

### Changed
- Improved code quality: ruff and mypy compliance across codebase
- Completed pattern descriptions (filled in missing documentation)
- Clarified architecture documentation for constrained-capacity model approach
- Documentation consistency improvements across the project
- Refined pattern reviewer agent prompts and workflow
- Reorganized pattern test file structure for consistency
- Updated evaluation framework with improved metrics handling

### Removed
- Removed pattern: `pt-002-missing-eval-mode` (merged functionality into `pt-007-inference-without-eval`)

## [0.1.0] - 2026-03-04

Initial public release.

### Features
- AI-powered detection of scientific Python code issues
- 44 detection patterns across 6 categories:
  - ML correctness and data leakage
  - PyTorch training bugs
  - Numerical precision issues
  - Reproducibility problems
  - Performance anti-patterns
  - Parallelization issues
- Local LLM integration via vLLM (Gemma 3 12B)
- Command-line interface with JSON and text output
- Support for Python scripts and Jupyter notebooks
- Evaluation framework with precision/recall metrics
- Designed for both human developers and AI coding agents

[0.1.5]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.5
[0.1.4]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.4
[0.1.3]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.3
[0.1.2]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.2
[0.1.1]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.1
[0.1.0]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.0
