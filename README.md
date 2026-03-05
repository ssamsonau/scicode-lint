# scicode-lint

**AI-powered linter for ML code written for scientific applications. Catches methodology bugs that traditional linters miss and that quietly invalidate your results.**

Runs locally on your GPU or institutional cluster. No cloud APIs, your code stays private, no unexpected GenAI bills.

---

## The Problem

Classifying disease from symptoms, analyzing EEG sleep patterns, predicting vaccination response, determining battery currents from magnetic fields, ranking molecular crystals, detecting pests from acoustics: researchers across science are now applying ML as a standard part of their work.

Traditional linters catch syntax errors and style issues. They can't catch methodology bugs: the kind where code runs, tests pass, and results are still wrong. Data leakage. Missing random seeds. Silent numerical errors.

And it's getting harder with AI coding tools. These tools are trained on public repositories full of methodology mistakes: Kaggle notebooks with data leakage, tutorials that skip random seeds, Stack Overflow answers with broken cross-validation. The bugs transfer seamlessly into your codebase.

**scicode-lint is designed to catch these.**

---

## Who This Is For

Scientists applying AI and ML to their research: biology, chemistry, physics, psychology, neuroscience, earth and materials sciences, linguistics, engineering, social science, and beyond.

You're an expert in your domain. Catching ML methodology bugs isn't your day job, and you may not know what to look for. This tool is built for that gap: automated checking that runs alongside your development workflow, the same way any linter would. Works interactively for individual researchers and integrates with AI coding agents and CI/CD pipelines.

---

## What It Does

Scans Python scripts and Jupyter notebooks for 44 initial patterns across six categories:

- **ai-training** (15 patterns): data leakage, PyTorch training modes, gradient management
- **ai-inference** (3 patterns): missing eval mode, missing no_grad, device mismatches
- **ai-data** (1 pattern): DataLoader configuration issues
- **scientific-numerical** (10 patterns): float comparison, dtype overflow, catastrophic cancellation
- **scientific-performance** (11 patterns): loops vs vectorization, memory inefficiency
- **scientific-reproducibility** (4 patterns): missing seeds, CUDA non-determinism

It tells you what's wrong and why it matters. No auto-fixes: you stay in control of your code.

### Example Output

```
test_scaler.py — 1 issue found

🔴 CRITICAL [method fit_transform] ml-001: Issue detected
   Data leakage: scaler is fit on full data including test set.
   Model performance will be inflated. Use sklearn.pipeline.Pipeline
   so fitting happens inside each fold.

   Code: X_scaled = scaler.fit_transform(X)
```

---

## Project Status

**Status: Work in Progress** (v0.1.0 alpha). Initial implementation and design.

**Completed:**
- 44 initial patterns implemented
- CLI working
- Test suites written for all patterns
- Evaluation framework in place

**Quality Metrics:**

*Pattern-specific tests* (one pattern per file, isolated):
- Precision: 82.5% (target: 90%)
- Recall: 84.6% (target: 80%) ✓
- Pass rate: 52% (23 of 44 patterns meet both thresholds)

*Integration tests* (realistic code, all patterns together):
- Precision: 24.2% (high false positive rate in practice)
- Recall: 50%
- Pass rate: 0/4 scenarios

Patterns perform well in isolation but generate significant false positives when run together on realistic code. This is the primary issue under active investigation. Using `--pattern` or `--category` flags to run specific patterns reduces noise considerably.

**In progress:** Reducing false positive rate, improving multi-pattern detection accuracy, fixing 5 patterns with known issues.

---

## Quick Start

### Prerequisites

- GPU with 20GB+ VRAM and native FP8 support (RTX 4090, RTX 4000 Ada, L4, etc.)
- vLLM 0.16+ with Gemma 3 12B FP8 model

See [INSTALLATION.md](INSTALLATION.md) for detailed setup.

### Installation

```bash
# Recommended: use pipx for isolation
pipx install scicode-lint[vllm-server]

# Start vLLM server
vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic --trust-remote-code --max-model-len 16000
```

### Usage

```bash
# Check a single file with one pattern (~5 seconds)
scicode-lint check my_pipeline.py --pattern ml-001

# Check Jupyter notebooks
scicode-lint check analysis.ipynb --pattern ml-001

# Check by category
scicode-lint check train.py --category ai-training

# Full scan, all 44 patterns (~30 seconds per file)
scicode-lint check train.py

# Filter by severity
scicode-lint check train.py --severity critical,high
```

---

## Current Limitations

- **High false positive rate when running all patterns together.** Recommended: use `--category` or `--pattern` flags to target specific checks rather than full scans until integration precision improves.
- **Single-file analysis only.** Issues that span multiple files (like preprocessing done differently in train.py and test.py) are out of scope for now.
- **Requires a GPU with 20GB+ VRAM.** Not practical for laptops or CPU-only setups. RTX 4090, RTX 4000 Ada, or L4 are the target hardware.

---

## How It Works

Runs a local LLM (Gemma 3 12B) via vLLM. Each pattern checks your code through a specific analysis lens: "does this code fit a scaler on data that includes the test set?", "is model.eval() called before inference?" Focused checks designed to catch real bugs.

Your code fits entirely in the model's context window (16K tokens ≈ 1,500 lines), so it reads the whole file for each check. All patterns run concurrently with automatic prefix caching.

See [docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) for technical details.

---

## Architecture

- **Model:** Gemma 3 12B FP8 via vLLM. Local-first design for privacy and zero API costs.
- **Prefix caching:** Code-first prompt structure means all 44 patterns share the same code prefix. vLLM caches the code once and reuses it across all pattern checks.
- **Concurrency:** All patterns run concurrently via async/await. vLLM's continuous batching processes them in parallel. Processing N patterns takes approximately the time of 1 pattern plus overhead.
- **Pattern structure:** Each pattern is a narrow, focused detection task defined in a TOML file with comprehensive test suites (positive/negative/context-dependent cases).
- **Evaluation:** Two-tier framework: pattern-specific tests in isolation, plus integration tests on realistic multi-pattern code. Target: 90% precision, 80% recall.

---

## Documentation

- 📖 [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md): find all documentation
- 🚀 [INSTALLATION.md](INSTALLATION.md): detailed setup guide
- 📘 [docs_use_human/USAGE.md](docs_use_human/USAGE.md): CLI usage guide
- 🤖 [docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md): for AI coding agents
- 🏗️ [CONTRIBUTING.md](CONTRIBUTING.md): adding new patterns
- 📊 [evals/README.md](evals/README.md): pattern evaluation framework
- 🔬 [evals/integration/README.md](evals/integration/README.md): integration tests

---

## Feedback Wanted

Early release. If you're a researcher applying ML to your domain:

- Which patterns are missing from your field?
- Which detections are too noisy to be useful?
- What would make this fit your actual workflow?

Open an issue or start a discussion on GitHub.

---

## Contributing

Each pattern lives in `patterns/{category}/{id}/` and needs:
- `pattern.toml`: the detection question and warning message
- Test files: examples of buggy code (positive), correct code (negative), and edge cases (context-dependent)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## License

MIT