# scicode-lint

**AI-powered linter for ML code written for scientific applications. Catches methodology bugs that traditional linters miss: the kind that quietly invalidate your results.**

Runs locally on your GPU or institutional cluster. No cloud APIs, your code stays private, no unexpected GenAI bills.

---

## TL;DR

Local LLM linter for scientific ML code. Catches data leakage, missing seeds, numerical bugs. Runs on your GPU (16GB+ VRAM), code stays private.

```bash
pip install scicode-lint                  # Use with remote vLLM server
# or
pip install scicode-lint[vllm-server]     # Run vLLM locally (16GB+ GPU)

scicode-lint check train.py               # Scan for issues
```

---

## The Problem

Traditional linters catch syntax errors and style issues. They can't catch methodology bugs: the kind where code runs, tests pass, and results are still wrong. Data leakage. Missing random seeds. Silent numerical errors.

And it's getting harder with AI coding tools. These tools are trained on public repositories full of methodology mistakes: Kaggle notebooks with data leakage, tutorials that skip random seeds, Stack Overflow answers with broken cross-validation. The bugs transfer seamlessly into your codebase.

**Built for scientists applying ML** - biology, chemistry, physics, neuroscience, engineering, and beyond. You're an expert in your domain; catching ML methodology bugs isn't your day job. This tool fills that gap.

---

## What It Does

Scans Python scripts and Jupyter notebooks for 66 patterns across five categories:

- **ai-training** (16 patterns): data leakage, PyTorch training modes, gradient management, DataLoader configuration
- **ai-inference** (13 patterns): missing eval mode, missing no_grad, device mismatches, CUDA timing, JIT tracing
- **scientific-numerical** (10 patterns): float comparison, dtype overflow, catastrophic cancellation
- **scientific-performance** (11 patterns): loops vs vectorization, memory inefficiency
- **scientific-reproducibility** (14 patterns): missing seeds, CUDA non-determinism, unsorted iteration, pickle versioning

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

## Quick Start

### Prerequisites

- GPU with 16GB+ VRAM and native FP8 support (RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, L4, etc.)
- vLLM 0.17+ with RedHatAI/Qwen3-8B-FP8-dynamic (default model)

See [INSTALLATION.md](INSTALLATION.md) for detailed setup.

### Installation

```bash
# With local vLLM server (runs on your GPU)
pip install scicode-lint[vllm-server]

# Or with remote vLLM server (e.g., university/institutional server)
pip install scicode-lint
scicode-lint check my_code.py --vllm-url https://vllm.your-institution.edu

# For development
git clone https://github.com/ssamsonau/scicode-lint.git
cd scicode-lint
pip install -e ".[all]"
```

### Usage

```bash
# Check a single file with one pattern
scicode-lint check my_pipeline.py --pattern ml-001

# Check Jupyter notebooks
scicode-lint check analysis.ipynb --pattern ml-001

# Check by category
scicode-lint check train.py --category ai-training

# Full scan, all 66 patterns
scicode-lint check train.py

# Filter by severity
scicode-lint check train.py --severity critical,high
```

---

## Current Limitations

- **Single-file analysis only.** Issues that span multiple files (like preprocessing done differently in train.py and test.py) are out of scope for now.
- **Requires a GPU with 16GB+ VRAM.** Not practical for laptops or CPU-only setups. RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, or L4 are the target hardware.

---

## Project Status

**Work in Progress** (v0.2.0 alpha)

| Test Type | Precision | Recall | Description |
|-----------|-----------|--------|-------------|
| Controlled tests | 99.5% | 99.5% | Curated positive/negative test files per pattern |
| Integration static | 84.6% | 89.2% | Pre-generated multi-pattern scenarios |
| Integration dynamic | 44.4% | 83.3% | Fresh LLM-generated code each run |
| Labeled Kaggle notebooks | 64% | 89% | Yang et al. ASE'22 dataset, human-labeled ground truth |
| Published papers | ~18%* | - | 38 AI+science papers; every paper had at least one verified bug |

*Pending re-run; precision expected to improve after bug fix (verified findings remain valid).

Example reports: [`real_world_demo/output_examples/`](real_world_demo/output_examples/)

---

## How It Works

**Design philosophy: Middle ground between grep and SOTA cloud reasoning.**

Traditional linters use grep-style pattern matching - fast but misses context. Cloud AI APIs offer deep reasoning but cost money, raise privacy concerns, and models get deprecated. scicode-lint uses a **local LLM with thinking mode** (RedHatAI/Qwen3-8B-FP8-dynamic, fits in 16GB VRAM via vLLM) that sits between these extremes:

- **Smarter than grep**: Understands code structure, follows data flow, catches semantic issues
- **Reasoning capability**: Uses thinking mode to analyze code behavior and intent, not just literal text
- **Local and private**: Your code never leaves your machine, no API costs
- **Reproducible**: Open-source models remain available; results stay consistent over time

**How patterns run:** Each pattern is a focused detection question in a TOML file. All 66 patterns run concurrently - vLLM's prefix caching means your code is processed once and shared across all checks. Processing N patterns takes approximately the time of 1 pattern.

**Design goal:** Patterns are grounded in official documentation (PyTorch docs, scikit-learn guides, NumPy API references). See [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) for technical details.

---

## Documentation

- 📖 [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md): find all documentation
- 🚀 [INSTALLATION.md](INSTALLATION.md): detailed setup guide
- 📘 [docs_use_human/USAGE.md](docs_use_human/USAGE.md): CLI usage guide
- 🤖 [docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md): for AI coding agents
- 🏗️ [CONTRIBUTING.md](CONTRIBUTING.md): adding new patterns
- 📊 [evals/README.md](evals/README.md): pattern evaluation framework
- 🔬 [evals/integration/README.md](evals/integration/README.md): integration tests
- 📈 [docs_dev_genai/CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md): pattern quality improvement workflow
- 📊 [docs_use_human/performance/VLLM_MONITORING.md](docs_use_human/performance/VLLM_MONITORING.md): vLLM monitoring dashboard
- 🌐 [real_world_demo/README.md](real_world_demo/README.md): real-world validation on scientific ML code from PapersWithCode repos and Yang et al. ASE'22 leakage paper (74% F1 on preprocessing leakage detection)

---

## Feedback Wanted

Early release. If you're a researcher applying ML to your domain:

- Which patterns are missing from your field?
- Which detections are too noisy to be useful?
- What would make this fit your actual workflow?

Open an issue or start a discussion on GitHub.

---

## Contributing

Each pattern lives in `src/scicode_lint/patterns/{category}/{id}/` and needs:
- `pattern.toml`: the detection question and warning message
- Test files: examples of buggy code (positive), correct code (negative), and edge cases (context-dependent)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Releasing

To create a new release (maintainers only):

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit and run: `./scripts/release.sh`

The script checks prerequisites, builds the package, creates a git tag, and publishes a GitHub release.

---

## Development Approach

GenAI-native development with Claude Code. Patterns are generated and iterated by AI agents within human-designed evaluation frameworks and quality gates. See [Continuous Improvement Loop](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md) for the workflow.

---

## License

MIT