# Documentation for GenAI Agents Working on scicode-lint

**Audience:** AI coding assistants contributing to or modifying the scicode-lint codebase

**Purpose:** Understand design principles, architecture, and implementation details

---

## Contents

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Core design principles and architectural decisions
  - Code-first prompts for prefix caching
  - Detection-only approach (no auto-fixes)
  - Narrow prompts per pitfall
  - Eval coverage requirements
  - Local-first LLM architecture

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical implementation details
  - Package structure
  - Core components (catalog, prompts, LLM client, formatter)
  - Async batching for pattern checks
  - Data models and API
  - Code quality standards

- **[CONTINUOUS_IMPROVEMENT.md](CONTINUOUS_IMPROVEMENT.md)** - Evaluation → improvement workflow

- **[../pattern_verification/README.md](../pattern_verification/README.md)** - Pattern quality verification
  - Deterministic checks (9 automated checks, no LLM)
  - Semantic checks (LLM-based consistency review)

---

## When to Use These Docs

Use these documents when:
- Adding new detection patterns
- Modifying the linter architecture
- Improving performance
- Refactoring code
- Understanding design decisions
- Contributing to the project

---

## Not for Package Users

If you're looking for how to **use** scicode-lint (not modify it), see:
- **[../docs_use_genai/](../docs_use_genai/)** - For AI agents using scicode-lint as a tool
- **[../README.md](../README.md)** - For human users
- **[../INSTALLATION.md](../INSTALLATION.md)** - Setup instructions
