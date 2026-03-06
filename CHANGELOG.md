# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.1
[0.1.0]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.0
