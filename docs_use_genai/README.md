# Documentation for GenAI Agents Using scicode-lint

**Audience:** AI coding assistants helping scientists write research code and using scicode-lint to check/fix it

**Purpose:** Learn how to integrate scicode-lint into coding workflows to detect and fix scientific code bugs

---

## Contents

- **[GENAI_AGENT_GUIDE.md](GENAI_AGENT_GUIDE.md)** - Complete guide for AI agents ⭐ START HERE
  - Installation and setup
  - Python API usage (recommended)
  - CLI usage (alternative)
  - Understanding results (which patterns failed, what they mean)
  - Targeted pattern checking (fast)
  - Complete workflow examples
  - Common patterns and fixes
  - Error handling

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API reference
  - All public classes and methods
  - Type signatures and parameters
  - Complete usage examples

- **[PATTERN_LOOKUP_EXAMPLE.md](PATTERN_LOOKUP_EXAMPLE.md)** - Pattern lookup API
  - Look up pattern details by ID
  - List all available patterns
  - Complete workflow: check → lookup → fix

- **[VLLM_UTILITIES.md](VLLM_UTILITIES.md)** - vLLM server management (optional)
  - Automated server start/stop for workflows
  - Context manager for automatic lifecycle
  - Useful for testing and CI/CD
  - NOT required for normal usage

- **[INTERFACE_ANALYSIS.md](INTERFACE_ANALYSIS.md)** - Package interface analysis
  - Is scicode-lint optimized for humans or AI agents?
  - What works well for GenAI agents
  - Recommendations for improvements

---

## Quick Start

### Option 1: Manual Server (Typical)

```python
# 1. Install and start server manually
pip install scicode-lint
vllm serve Qwen/Qwen3-8B-FP8 --trust-remote-code --max-model-len 20000

# 2. Use in code
from scicode_lint import SciCodeLinter
linter = SciCodeLinter()
result = linter.check_file(Path("myfile.py"))

# 3. Fix issues
for finding in result.findings:
    print(f"{finding.id}: {finding.explanation}")
    apply_fix(file_path, finding)
```

### Option 2: Automated Server (GenAI Workflows)

```python
# Install
pip install scicode-lint[vllm-server]

# Auto-managed server lifecycle
from scicode_lint.vllm import VLLMServer

with VLLMServer():  # Server starts automatically
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
    for finding in result.findings:
        apply_fix(finding)
# Server stops automatically
```

See **[GENAI_AGENT_GUIDE.md](GENAI_AGENT_GUIDE.md)** for complete documentation.

---

## Use Case

**Typical Workflow:**
1. Scientist writes scientific Python code with AI coding assistant
2. AI agent runs scicode-lint to check for common bugs (data leakage, missing seeds, etc.)
3. AI agent fixes issues based on finding explanations
4. AI agent verifies fixes by re-running linter

---

## Not for scicode-lint Contributors

If you're looking to **modify** scicode-lint (not use it), see:
- **[../docs_dev_genai/](../docs_dev_genai/)** - For AI agents working on the codebase
