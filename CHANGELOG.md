# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-04

Initial public release.

### Features
- AI-powered detection of scientific Python code issues
- 44 detection patterns across 8 categories:
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

[0.1.0]: https://github.com/ssamsonau/scicode-lint/releases/tag/v0.1.0
