# Pattern Verification

Tools for verifying pattern quality. Two complementary approaches:

## Deterministic Checks (no LLM needed)

Fast, automated checks that catch structural issues:

```bash
# Check all patterns
python pattern_verification/deterministic/validate.py

# Check specific pattern
python pattern_verification/deterministic/validate.py ml-001-scaler-leakage

# Auto-fix what's possible
python pattern_verification/deterministic/validate.py --fix

# Fail on warnings too
python pattern_verification/deterministic/validate.py --strict
```

**9 checks performed:**

| Check | Type | Description |
|-------|------|-------------|
| TOML/file sync | error | Every test file has TOML entry and vice versa |
| Schema validation | error | pattern.toml matches Pydantic model |
| Data leakage hints | error | No `# BUG:`, `# CORRECT:` comments in test files |
| Test file count | warning | Minimum 3 positive, 3 negative recommended |
| TODO markers | error | No unfinished placeholders in TOML |
| Detection format | error | Question ends with YES/NO conditions |
| Test file syntax | error | All .py files are valid Python |
| Empty fields | error | Required fields have content |
| Test diversity | warning | Detect copy-paste via AST similarity |

**Auto-fix capabilities:**
- Add missing TOML entries for test files on disk
- Remove orphaned TOML entries (files that don't exist)
- Rename entries when close filename match exists

## Semantic Checks (uses LLM)

Deep review that catches consistency issues scripts can't detect.

### ⚠️ COST WARNING - IMPORTANT

**THIS USES CLAUDE CODE AGENTS AND CONSUMES YOUR TOKENS QUICKLY!**

Each pattern review spawns a Claude Code agent (Opus model) that reads and analyzes files. This costs tokens from your Claude subscription or API quota.

Before running semantic validation, CONFIRM with user:
- **Single pattern?** OK to proceed.
- **Selected few patterns?** OK if deterministic checks pass first.
- **ALL 64 patterns?** CONFIRM with user first - this is expensive!

When asked to review patterns:
1. Always run deterministic checks first (free, no tokens)
2. Ask user: "Run semantic validation on selected patterns or all 64?"
3. Only proceed with bulk review after explicit user confirmation

**Token costs:**
- Each pattern review reads pattern.toml + all test files
- Reviewing all 64 patterns in parallel multiplies token usage significantly
- Always run deterministic checks first - they're free and catch most issues

### Usage

```bash
# Review single pattern (recommended)
claude --agent pattern-reviewer "Review ml-001-scaler-leakage"

# Review category (use sparingly)
claude --agent pattern-reviewer "Review all ai-inference patterns"
```

**Note:** The agent is symlinked from `pattern_verification/semantic/pattern-reviewer/` to `.claude/agents/pattern-reviewer` for better parallel execution support. Edit the source at `pattern_verification/semantic/pattern-reviewer/`.

**Model:** Uses Opus (configured in YAML frontmatter). Do NOT use Haiku - it lacks reasoning capability for consistency verification.

### What it checks

- Does `description` accurately describe the test file code?
- Does `expected_issue` align with detection question's YES condition?
- Does `snippet` actually exist in the test file?
- Do positive tests actually contain the bug described?
- Do negative tests actually avoid the bug?

### When to use

- After auto-generating descriptions (especially if generated with cheaper models)
- When evals fail unexpectedly
- Before merging new patterns
- **NOT for routine checks** - use deterministic validation instead

## Recommended Workflow

```bash
# 1. Run deterministic checks first (fast, catches obvious issues)
python pattern_verification/deterministic/validate.py

# 2. Fix any errors
python pattern_verification/deterministic/validate.py --fix

# 3. Run semantic review (slower, catches consistency issues)
claude --agent pattern_verification/semantic/pattern-reviewer "Review <pattern-id>"

# 4. Run evals (requires vLLM server)
python evals/run_eval.py --pattern <pattern-id>
```

## Directory Structure

```
pattern_verification/
├── deterministic/
│   └── validate.py          # 9 automated checks
├── semantic/
│   └── pattern-reviewer/    # Agent source (symlinked to .claude/agents/)
│       ├── agent.json
│       └── system_prompt.md
└── README.md

.claude/agents/
└── pattern-reviewer -> ../../pattern_verification/semantic/pattern-reviewer
```
