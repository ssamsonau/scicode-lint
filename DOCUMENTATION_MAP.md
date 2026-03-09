# Documentation Map

Quick guide to find the right documentation for your needs.

---

## For Human Users

**Getting Started:**
- [README.md](README.md) - Overview, quick start, features
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation instructions
- [docs_use_human/USAGE.md](docs_use_human/USAGE.md) - Usage guide

**Setup Guides:**
- [INSTALLATION.md](INSTALLATION.md) - Installation and vLLM setup
- [docs_use_human/VRAM_REQUIREMENTS.md](docs_use_human/VRAM_REQUIREMENTS.md) - VRAM requirements and model selection

**Performance:**
- [docs_use_human/performance/BENCHMARKING.md](docs_use_human/performance/BENCHMARKING.md) - Benchmarking guide
- [docs_use_human/performance/CONCURRENCY_GUIDE.md](docs_use_human/performance/CONCURRENCY_GUIDE.md) - Concurrency optimization
- [docs_use_human/performance/VLLM_MONITORING.md](docs_use_human/performance/VLLM_MONITORING.md) - vLLM monitoring and dashboard

**Contributing:**
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## For GenAI Agents USING scicode-lint

**Use Case:** AI coding assistant helping scientist write code → uses scicode-lint to check/fix bugs

**Documentation:**
- **[docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md)** ⭐ START HERE
  - Complete guide for integrating scicode-lint
  - Python API and CLI usage
  - Understanding results
  - Targeted checking
  - Workflow examples
  - Common fixes

- [docs_use_genai/INTERFACE_ANALYSIS.md](docs_use_genai/INTERFACE_ANALYSIS.md) - Package interface analysis

---

## For GenAI Agents WORKING ON scicode-lint

**Use Case:** AI assistant modifying/contributing to the scicode-lint codebase

**Documentation:**
- **[docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)** - Design principles
- **[docs_dev_genai/IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md)** - Technical implementation
- [CLAUDE.md](CLAUDE.md) - Instructions for Claude Code CLI

**Pattern verification:**
- **[pattern_verification/](pattern_verification/)** - Deterministic and semantic quality checks
  - `deterministic/validate.py` - 9 automated checks (no LLM needed)
  - `semantic/pattern-reviewer/` - LLM-based consistency checking

---

## Directory Structure

```
/
├── README.md                       # Main readme (humans)
├── INSTALLATION.md                 # Setup guide (humans)
├── DOCUMENTATION_MAP.md            # This file
│
├── docs_use_human/                 # Human user documentation
│   ├── USAGE.md
│   ├── VRAM_REQUIREMENTS.md
│   └── performance/
│       ├── BENCHMARKING.md
│       ├── CONCURRENCY_GUIDE.md
│       └── VLLM_MONITORING.md
│
├── docs_use_genai/               # GenAI agents USING scicode-lint
│   ├── README.md
│   ├── GENAI_AGENT_GUIDE.md       ⭐ Main guide
│   └── INTERFACE_ANALYSIS.md
│
├── docs_dev_genai/                 # GenAI agents WORKING ON scicode-lint
│   ├── README.md
│   ├── ARCHITECTURE.md
│   └── IMPLEMENTATION.md
│
├── pattern_verification/           # Pattern quality verification
│   ├── deterministic/validate.py   # 9 automated checks
│   └── semantic/pattern-reviewer/  # LLM-based consistency checking
│
├── patterns/                       # Pattern definitions and tests
│   ├── README.md                  # Pattern guide (structure, format, detection question template)
│   └── {category}/{pattern}/       # Individual pattern directories
│
├── evals/                          # Evaluation framework
│   ├── README.md                   # Pattern-specific evaluations
│   ├── run_eval.py                 # Hardcoded ground truth
│   ├── run_eval.py       # LLM-as-judge
│   └── integration/                # Multi-pattern integration tests
│       ├── README.md
│       ├── run_integration_eval.py # Hardcoded ground truth
│       └── run_integration_eval_llm_judge.py  # LLM-as-judge
│
├── benchmarks/                     # Performance benchmarks
│   └── max_tokens_experiment.py    # Token limit tuning
│
├── tools/                          # Development and monitoring tools
│   ├── vllm_dashboard.py           # Streamlit monitoring dashboard
│   └── start_dashboard.sh          # Dashboard launcher script
│
└── scripts/                        # Utility scripts
    └── project_stats_generate.py   # Project statistics generator (--help for usage)
```

---

## Quick Links by Task

### "I want to use scicode-lint to check my code"
→ [README.md](README.md) → [INSTALLATION.md](INSTALLATION.md) → [docs_use_human/USAGE.md](docs_use_human/USAGE.md)

### "I'm a GenAI agent helping a scientist write code"
→ [docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md)

### "I want to contribute a new detection pattern"
→ [CONTRIBUTING.md](CONTRIBUTING.md) → [docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)

### "I want to understand the architecture"
→ [docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) → [docs_dev_genai/IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md)

### "I want to optimize performance"
→ [docs_use_human/performance/CONCURRENCY_GUIDE.md](docs_use_human/performance/CONCURRENCY_GUIDE.md)

### "I want to monitor vLLM during evals"
→ [docs_use_human/performance/VLLM_MONITORING.md](docs_use_human/performance/VLLM_MONITORING.md)

### "I want to benchmark scicode-lint"
→ [docs_use_human/performance/BENCHMARKING.md](docs_use_human/performance/BENCHMARKING.md)

### "I want to review or improve pattern definitions"
→ [pattern_verification/README.md](pattern_verification/README.md) → [CONTRIBUTING.md](CONTRIBUTING.md)

### "I want to run evaluations"
→ [evals/README.md](evals/README.md) (pattern-specific) → [evals/integration/README.md](evals/integration/README.md) (multi-pattern)

### "I need project statistics"
→ Run `python scripts/project_stats_generate.py --help`
