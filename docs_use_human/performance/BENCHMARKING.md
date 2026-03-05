# Benchmarking Guide

This guide explains how to benchmark scicode-lint performance on your system.

## Quick Start

```bash
# Install dependencies first
pip install -e .

# Make sure vLLM backend is running

# Main benchmark (5 patterns, ~10 seconds) - RECOMMENDED
python benchmarks/benchmark.py

# Full system benchmark (44 patterns on 20 files, ~3 minutes)
python benchmarks/benchmark_system.py

# See benchmarks/README.md for details
```

## What Gets Measured

System information, LLM backend details, and performance metrics (total time, average per file, throughput).

## Output Files

Results are saved to two files (both ignored by git):

1. **benchmark_results.json** - Machine-readable JSON format
2. **benchmark_results.txt** - Human-readable text format

## Using Verbosity Flags

The CLI supports verbosity levels: `-v` (INFO), `-vv` (DEBUG), `-vvv` (TRACE). Use `--benchmark` flag for detailed performance summary.

## Example Output

```
============================================================
BENCHMARK SUMMARY
============================================================
Total files checked: 20
Total findings: 45
Total time: 67.84s
Average time per file: 3.39s
Files per minute: 17.7
============================================================
```

## Interpreting Results

**Expected Performance:**
- Modern GPU: 2-5s per file
- Older GPU: 5-10s per file

## Optimizing Performance

1. **Filter severities**: Use `--severity critical,high` to skip medium-priority checks
2. **Model selection**: Smaller models are faster than larger ones

## Sharing Results

Include benchmark_results.txt, LLM backend/model info, and hardware specs when reporting performance.

## Available Benchmarks

See the `benchmarks/` directory for performance testing:

- **`benchmark.py`** - Quick 10-second test (sequential vs concurrent, 5 patterns)
- **`benchmark_system.py`** - Full system benchmark (20 files, 44 patterns, ~3 minutes)

See `benchmarks/README.md` for detailed documentation.
