# scicode-lint Usage Guide

## Installation

```bash
pip install -e .
```

## Prerequisites

The linter requires a local LLM server to run. vLLM is the only supported backend:

### vLLM Server Setup

```bash
# Install vLLM server
pip install scicode-lint[vllm-server]

# Start vLLM server (downloads model automatically on first run)
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic \
    --trust-remote-code --gpu-memory-utilization 0.85 \
    --max-model-len 20000

# For CPU-only systems, add --device cpu
# Note: CPU inference is 10-50x slower than GPU
```

See [INSTALLATION.md](../../INSTALLATION.md) for detailed vLLM configuration and [VRAM_REQUIREMENTS.md](VRAM_REQUIREMENTS.md) for hardware requirements.

## Basic Usage

### Check a single file

```bash
scicode-lint check myfile.py
```

or

```bash
python -m scicode_lint check myfile.py
```

### Check multiple files

```bash
scicode-lint check file1.py file2.py notebook.ipynb
```

### Check a directory (recursive)

```bash
scicode-lint check src/
```

## Output Formats

### Text output (default)

Human-readable output with severity indicators.

### JSON output

Use `--format json` for machine-parseable output.

### JSON output with error reporting

For GenAI agents and automated workflows, use `--json-errors` to include errors in structured format:

```bash
scicode-lint check myfile.py --format json --json-errors
```

When errors occur (e.g., file too large), they're included in the output:

```json
[
  {
    "file": "large_file.py",
    "findings": [],
    "error": {
      "file": "large_file.py",
      "error_type": "ContextLengthError",
      "message": "File too large for context window\n  File: large_file.py\n  ...",
      "details": {
        "error": "ContextLengthError",
        "file_path": "large_file.py",
        "estimated_tokens": 12000,
        "max_tokens": 8000,
        "overflow": 4000,
        "suggestions": [
          "Split into smaller files (< 8,000 tokens)",
          "Focus linting on specific functions/classes",
          "Increase max_model_len when starting vLLM server",
          "Use a model with larger context window"
        ]
      }
    },
    "summary": {
      "total_findings": 0,
      "by_severity": {},
      "by_category": {}
    }
  }
]
```

**Use cases:**
- **Automated pipelines**: Parse errors programmatically
- **GenAI agents**: Structured error handling in AI workflows
- **Logging systems**: Forward structured errors to monitoring tools

## Filtering

### By severity

```bash
# Only critical issues
scicode-lint check myfile.py --severity critical

# Critical and high
scicode-lint check myfile.py --severity critical,high

# All levels (default)
scicode-lint check myfile.py --severity critical,high,medium
```

### By confidence

```bash
# Only high-confidence findings (default: 0.7)
scicode-lint check myfile.py --min-confidence 0.85
```

## Server Configuration

### Default (auto-detect)

```bash
# Auto-detects vLLM on ports 5001 or 8000
scicode-lint check myfile.py
```

### Custom vLLM server

```bash
# Specify custom URL and model
scicode-lint check myfile.py \
  --vllm-url http://localhost:8000 \
  --model RedHatAI/Qwen3-8B-FP8-dynamic
```

## Exit Codes

- `0`: No issues found
- `1`: Issues found or error occurred

## Performance Tips

1. **vLLM with prefix caching** enables fast checking of multiple patterns on the same file
2. **Reduce severity levels** to check only critical/high issues if needed
3. **Increase confidence threshold** to reduce false positives

## Detection Categories

The linter checks 66 patterns across these categories:

- **ai-training** (16 patterns) - Data leakage, PyTorch training modes, gradient management
- **ai-inference** (13 patterns) - Missing eval mode, missing no_grad, device mismatches
- **scientific-numerical** (10 patterns) - Float comparison, dtype overflow, catastrophic cancellation
- **scientific-performance** (11 patterns) - Loops vs vectorization, memory inefficiency
- **scientific-reproducibility** (14 patterns) - Missing seeds, CUDA non-determinism

See the `patterns/` directory for complete list.

## Limitations

1. **Requires LLM**: vLLM must be running
2. **Speed**: Checking 66 patterns takes time (varies by file size and GPU)
3. **False positives possible**: Review all findings - the linter is conservative but not perfect
4. **Function/class level only**: Findings report function/class names, not exact line numbers
5. **Context length limits**: Files must fit within model's input context (16K tokens = ~1,500 lines)
   - Files that are too large will be skipped with a clear error message
   - See "File too large" error below for solutions

## Troubleshooting

### "Connection refused" error

Make sure vLLM is running:
```bash
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic
```

### "Model not found" error

The model should download automatically on first run. If issues persist, pre-download using HuggingFace's `snapshot_download`.

### Slow performance

- Filter to only critical/high severity
- Use smaller context window if memory is limited

### "File too large for context window" error

The linter validates file size before sending to vLLM. If a file is too large, you'll see:

```
File too large for context window
  File: large_module.py
  Estimated tokens: 10,000
  Context limit: 8,000
  Overflow: 2,000 tokens

Suggestions:
  • Split into smaller files (< 8,000 tokens)
  • Focus linting on specific functions/classes
  • Increase max_model_len when starting vLLM server
  • Use a model with larger context window
```

**Solutions:**

1. **Split the file**: Break large modules into smaller, focused files
   ```bash
   # Good: Multiple focused files
   src/preprocessing.py   (2000 tokens)
   src/training.py       (3000 tokens)
   src/evaluation.py     (2000 tokens)
   ```

2. **Adjust context limit**: The default 20K context (16K input + 4K response) supports ~1,500 line files (90-95th percentile):
   ```bash
   # Standard: 20K total tokens (16K input + 4K response)
   vllm serve RedHatAI/Qwen3-8B-FP8-dynamic --max-model-len 20000
   ```

3. **Use environment variable**: Override context limit via config
   ```bash
   export SCICODE_LINT_MAX_MODEL_LEN=20000
   scicode-lint check large_file.py
   ```

**Context window:**
- Standard: 20K tokens total (16K input + 4K response, 16GB VRAM, covers ~1,500 lines)

**Estimate tokens before checking:**
```python
from scicode_lint.llm import estimate_tokens
from pathlib import Path

tokens = estimate_tokens(Path("myfile.py").read_text())
print(f"Estimated tokens: {tokens:,}")
```

## Integration with CI/CD

```bash
# In GitHub Actions, GitLab CI, etc.
scicode-lint check src/ --severity critical --format json > findings.json

# Exit code is 1 if issues found, so it will fail the build
```
