# scicode-lint Guide for GenAI Coding Agents

**Purpose:** Enable AI coding assistants to check scientific Python code for bugs and fix them

**Quick Start:** Install → Check → Parse Results → Fix Issues

---

## Quick Reference

```python
# 1. Install
pip install scicode-lint

# 2. Setup backend (one-time)
pip install scicode-lint[vllm-server]

# 3. Use in your code
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig

linter = SciCodeLinter()
result = linter.check_file(Path("myfile.py"))

# 4. Know which patterns failed and what they mean
for finding in result.findings:
    print(f"{finding.id}: {finding.explanation}")  # WHICH & WHAT
    print(f"Location: {finding.location.name}")     # WHERE
    print(f"Code: {finding.location.snippet}")      # EXACT CODE

    # Look up full pattern details if needed
    pattern = linter.get_pattern(finding.id)
    print(f"Category: {pattern.category}")

# 5. List available patterns
for pattern in linter.list_patterns():
    print(f"{pattern.id}: {pattern.warning_message[:50]}")

# 6. Check specific patterns only (fast)
config = LinterConfig(enabled_patterns={"ml-001"})  # 4 sec, not 2 min
linter = SciCodeLinter(config)
```

---

## Installation

**Isolated Environment Recommended:** See [installation guide](../INSTALLATION.md) for setup options.

**Safety Note:** scicode-lint only reads code files as text - it never executes or imports your code.

```bash
# Install
pip install scicode-lint[vllm-server]

# Start vLLM server (see installation guide for configuration options)
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code --max-model-len 20000
```

**For automated workflows:** See [VLLM_UTILITIES.md](VLLM_UTILITIES.md) for programmatic server management.


**HPC Users:** Use institutional vLLM servers or dedicated inference nodes (L4, A10) only. See main installation guide for details.

---

## Python API (Recommended)

### Basic Usage

```python
from pathlib import Path
from scicode_lint import SciCodeLinter

# Check file
linter = SciCodeLinter()
result = linter.check_file(Path("myfile.py"))

# Access results
for finding in result.findings:
    print(f"{finding.severity} | {finding.id} | {finding.location.name}")
    print(f"Issue: {finding.explanation}")
    print(f"Code: {finding.location.snippet}\n")
```

### Configuration

```python
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig, LLMConfig, Severity

config = LinterConfig(
    llm_config=LLMConfig(
        base_url="http://localhost:5001",  # Auto-detects if empty
        model="RedHatAI/Qwen3-8B-FP8-dynamic",  # Auto-detects if empty
    ),
    min_confidence=0.7,  # 0.0-1.0
    enabled_severities={Severity.CRITICAL, Severity.HIGH},  # Filter by severity
    enabled_categories={"ai-training", "ai-inference"},  # Filter by category
    enabled_patterns={"ml-001", "pt-001"},  # Filter by pattern ID
)

linter = SciCodeLinter(config)
```

### Result Structure

```python
# LintResult
result.file: Path              # File that was checked
result.findings: list[Finding] # List of issues found
result.summary: dict           # Statistics

# Finding (tells you which pattern failed and why)
finding.id: str               # Pattern ID (e.g., "ml-001") ← WHICH pattern
finding.category: str         # "ai-training", "ai-inference", etc.
finding.severity: str         # "critical", "high", "medium"
finding.explanation: str      # What's wrong and how to fix ← WHAT it means
finding.confidence: float     # 0.0 to 1.0
finding.location.location_type: str  # "function", "class", "method", "module"
finding.location.name: str           # Function/class/method name ← WHERE
finding.location.lines: list[int]    # Full line range of function/method
finding.location.focus_line: int     # Specific line to look at ← EXACT LINE
finding.location.snippet: str        # Code snippet ← EXACT CODE
```

### Reading Pattern Descriptions

**Method 1: Via SciCodeLinter (Recommended - Simple)**

```python
from scicode_lint import SciCodeLinter

linter = SciCodeLinter()

# Get specific pattern by ID
pattern = linter.get_pattern("ml-001")
if pattern:
    print(f"Pattern: {pattern.id}")
    print(f"Category: {pattern.category}")
    print(f"Severity: {pattern.severity}")
    print(f"What it checks: {pattern.detection_question}")
    print(f"Why it matters: {pattern.warning_message}")

# List all available patterns
for pattern in linter.list_patterns():
    print(f"{pattern.id} ({pattern.severity}): {pattern.warning_message[:50]}...")
```

**Method 2: Via DetectionCatalog (Advanced)**

```python
from scicode_lint import DetectionCatalog
from scicode_lint.config import Severity
from pathlib import Path

# Load catalog directly
catalog_path = Path(__file__).parent / "scicode_lint" / "detection_catalog.yaml"
catalog = DetectionCatalog(catalog_path)

# Get patterns by category
ml_patterns = catalog.get_patterns_by_category("ai-training")
for p in ml_patterns:
    print(f"{p.id}: {p.warning_message}")

# Get patterns by severity
critical = catalog.get_patterns_by_severity(Severity.CRITICAL)
print(f"Found {len(critical)} critical patterns")
```

**Why use this:** Know what patterns exist before running checks, understand what issues mean, or display pattern info to users.

---

## CLI Usage

```bash
# Basic check
scicode-lint lint myfile.py

# JSON output (for parsing)
scicode-lint lint myfile.py --format json

# Filter by severity
scicode-lint lint myfile.py --severity critical,high

# Filter by category
scicode-lint lint myfile.py --category ai-training,ai-inference

# Filter by pattern
scicode-lint lint myfile.py --pattern ml-001,ml-002

# Check directory
scicode-lint lint src/

# Multiple files
scicode-lint lint file1.py file2.py
```

### JSON Output Format

```json
[
  {
    "file": "myfile.py",
    "findings": [
      {
        "id": "ml-001",
        "category": "ai-training",
        "severity": "critical",
        "location": {
          "type": "function",
          "name": "preprocess_data",
          "snippet": "scaler.fit_transform(X)"
        },
        "explanation": "Data leakage: scaler/encoder is fit on full data...",
        "confidence": 0.92
      }
    ],
    "summary": {
      "total_findings": 1,
      "by_severity": {"critical": 1},
      "by_category": {"ai-training": 1}
    }
  }
]
```

---

## AI Agent Workflow

### Complete Example: Fix Specific File & Patterns

```python
from pathlib import Path
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig

def fix_ml_pipeline(file_path: str):
    """
    Example: GenAI agent fixing ML pipeline code.
    Focus on ML correctness patterns only.
    """

    # Step 1: Check ONLY ML correctness issues (not all 66 patterns)
    config = LinterConfig(
        enabled_categories={"ai-training"},  # 19 patterns
    )
    linter = SciCodeLinter(config)
    result = linter.check_file(Path(file_path))

    if not result.findings:
        print(f"✓ {file_path}: No ML correctness issues")
        return

    # Step 2: Fix each issue
    for finding in result.findings:
        print(f"\nFixing {finding.id} in {finding.location.name}")
        print(f"  Issue: {finding.explanation}")

        if finding.id == "ml-001":
            # Data leakage: move scaler after train_test_split
            fix_scaler_leakage(file_path, finding)

        elif finding.id == "ml-004":
            # Wrong metric: replace accuracy with AUC-ROC
            fix_metric(file_path, finding)

        # Add more fixes as needed...

    # Step 3: Verify fixes (check same patterns again)
    print("\n=== Verifying fixes ===")
    result = linter.check_file(Path(file_path))
    if result.findings:
        print(f"⚠ Still has {len(result.findings)} issues")
    else:
        print(f"✓ All ML correctness issues fixed")

def fix_pytorch_training(file_path: str):
    """
    Example: GenAI agent fixing PyTorch training loop.
    Focus on PyTorch patterns only.
    """

    # Check ONLY PyTorch patterns
    config = LinterConfig(
        enabled_categories={"ai-training"},  # 19 patterns
    )
    linter = SciCodeLinter(config)
    result = linter.check_file(Path(file_path))

    for finding in result.findings:
        if finding.id == "pt-001":
            add_zero_grad(file_path, finding)
        elif finding.id == "pt-003":
            add_no_grad_context(file_path, finding)

def fix_single_issue(file_path: str, pattern_id: str):
    """
    Example: GenAI agent fixing ONE specific issue.
    Check only that pattern.
    """

    # Check ONLY the pattern you're fixing (fastest)
    config = LinterConfig(
        enabled_patterns={pattern_id},  # Single pattern, ~4-5 sec
    )
    linter = SciCodeLinter(config)
    result = linter.check_file(Path(file_path))

    if result.findings:
        finding = result.findings[0]
        print(f"Fixing {finding.id}: {finding.explanation}")
        apply_fix(file_path, finding)

        # Verify
        result = linter.check_file(Path(file_path))
        assert len(result.findings) == 0, "Fix failed"
        print(f"✓ {pattern_id} fixed")

# Usage examples
if __name__ == "__main__":
    # Scenario 1: Fixing ML pipeline
    fix_ml_pipeline("data_pipeline.py")

    # Scenario 2: Fixing PyTorch training
    fix_pytorch_training("train.py")

    # Scenario 3: Fixing specific issue the user reported
    fix_single_issue("preprocess.py", "ml-001")
```

### Workflow Steps

1. **Check**: `result = linter.check_file(Path(file_path))`
2. **Iterate**: `for finding in result.findings`
3. **Fix**: Use `finding.explanation` to understand what to change
4. **Verify**: Re-run linter to confirm fix

### Targeted Checking (Recommended During Development)

When fixing specific issues, check ONLY those patterns to save time:

```python
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig

# Example 1: GenAI fixing data leakage - check only ml-001
config = LinterConfig(
    enabled_patterns={"ml-001"},  # Only this pattern
)
linter = SciCodeLinter(config)
result = linter.check_file(Path("pipeline.py"))  # ~50 seconds

# Example 2: GenAI fixing PyTorch training loop - check only pt-001, pt-002, pt-003
config = LinterConfig(
    enabled_patterns={"pt-001", "pt-002", "pt-003"},
)
linter = SciCodeLinter(config)
result = linter.check_file(Path("train.py"))  # ~55 seconds

# Example 3: Check specific category while working on ML pipeline
config = LinterConfig(
    enabled_categories={"ai-training"},  # 19 patterns
)
linter = SciCodeLinter(config)
result = linter.check_file(Path("ml_pipeline.py"))  # ~60 seconds
```

**CLI equivalent:**
```bash
# Check only ml-001 while fixing data leakage
scicode-lint lint pipeline.py --pattern ml-001

# Check multiple specific patterns
scicode-lint lint train.py --pattern pt-001,pt-002,pt-003

# Check category
scicode-lint lint ml_pipeline.py --category ai-training
```

**Best Practice:** During development, check only the patterns you're actively fixing. Run full scan before final commit.

### Understanding Which Patterns Failed & What They Mean

When you run a check, each `Finding` tells you exactly what failed:

```python
from scicode_lint import SciCodeLinter
from pathlib import Path

linter = SciCodeLinter()
result = linter.check_file(Path("pipeline.py"))

# Check which patterns failed
for finding in result.findings:
    # finding.id tells you WHICH pattern failed
    pattern_id = finding.id  # e.g., "ml-001"

    # finding.explanation tells you WHAT it means and HOW to fix
    what_and_how = finding.explanation
    # e.g., "Data leakage: scaler/encoder is fit on full data including test set.
    #        Model performance will be inflated. Use sklearn.pipeline.Pipeline so
    #        fitting happens inside each fold."

    # finding.location tells you WHERE in the code
    where = finding.location.name  # e.g., "preprocess_data"
    exact_code = finding.location.snippet  # e.g., "scaler.fit_transform(X)"

    print(f"Pattern {pattern_id} failed in {where}:")
    print(f"  Problem: {what_and_how}")
    print(f"  Code: {exact_code}")

    # OPTIONAL: Look up full pattern details
    pattern = linter.get_pattern(pattern_id)
    if pattern:
        print(f"  Category: {pattern.category}")
        print(f"  Severity: {pattern.severity}")
        print(f"  What it checks: {pattern.detection_question}")
```

**Example Output:**
```
Pattern ml-001 failed in preprocess_data:
  Problem: Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.
  Code: scaler.fit_transform(X)

Pattern pt-001 failed in train_loop:
  Problem: Missing optimizer.zero_grad(): gradients accumulate across batches. Loss will explode, then NaN. Call zero_grad() before each backward().
  Code: loss.backward()
```

**Summary:**
- `finding.id` → Which pattern (e.g., "ml-001", "pt-001")
- `finding.explanation` → What it means & how to fix
- `finding.location.name` → Where (function/class name)
- `finding.location.snippet` → Exact code line

All information needed to fix the issue is in the `Finding` object.

---

## Detection Categories (66 patterns)

| Category | Patterns | Examples |
|----------|----------|----------|
| **ai-training** | 19 | Data leakage, missing zero_grad, gradient issues |
| **ai-inference** | 12 | Missing eval mode, no_grad, device mismatches |
| **scientific-numerical** | 10 | Float comparison, overflow, division by zero |
| **scientific-performance** | 11 | Loops vs vectorization, memory inefficiency |
| **scientific-reproducibility** | 14 | Missing seeds, CUDA non-determinism |

### Common Patterns & Fixes

#### ml-001: Data Leakage (Scaler on Full Data)

**Problem:** Scaler fit on data before train/test split
```python
# Bad
scaler.fit_transform(X)
X_train, X_test = train_test_split(X)
```

**Fix:** Fit only on training data
```python
# Good
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform, not fit_transform
```

#### pt-001: Missing zero_grad()

**Problem:** Gradients accumulate across batches
```python
# Bad
for batch in dataloader:
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
```

**Fix:** Clear gradients before backward
```python
# Good
for batch in dataloader:
    optimizer.zero_grad()  # Add this
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
```

#### pt-003: Missing torch.no_grad()

**Problem:** Memory buildup during inference
```python
# Bad
model.eval()
predictions = model(X_test)
```

**Fix:** Disable gradient computation
```python
# Good
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

---

## Error Handling

```python
from scicode_lint import SciCodeLinter

try:
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
except FileNotFoundError:
    print("File not found")
except ConnectionError:
    print("LLM backend not running")
except Exception as e:
    print(f"Error: {e}")
```

### Common Issues

| Error | Solution |
|-------|----------|
| Connection refused | Start vLLM server |
| Model not found | Model downloads automatically on first run |
| Timeout | Increase timeout or check fewer patterns |
| Out of memory | Reduce `gpu-memory-utilization` in vLLM config |

---

## Performance

**Speed (per file):**
- Single pattern: ~50 seconds
- Category (8 patterns): ~60 seconds
- Full scan (66 patterns): ~90 seconds

**Optimization:**
- vLLM's prefix caching reuses code analysis across patterns for significant speedup
- Filter by `--severity critical` for faster checks
- Filter by `--category` for targeted analysis

**Hardware:**
- Tested on NVIDIA RTX 4000 Ada (20GB VRAM) with Qwen3-8B-FP8 @ 20K context
- Minimum: 16GB VRAM with native FP8 support (compute capability >= 8.9)

---

## Configuration

### Environment Variables

Optional environment variables: `OPENAI_BASE_URL`, `SCICODE_LINT_TEMPERATURE`, `SCICODE_LINT_TIMEOUT`.

All patterns are checked concurrently - vLLM handles batching internally.

### Config File

Create `~/.config/scicode-lint/config.toml`:

```toml
[llm]
# base_url = "http://localhost:5001"  # Optional, auto-detects if not set
temperature = 0.3

[linter]
min_confidence = 0.7
enabled_severities = ["critical", "high"]
```

---

## Limitations

1. **Single-file analysis**: Cross-file issues not detected
2. **Name-based detection**: LLM identifies function/class names, AST resolves to line numbers
   - More reliable than direct LLM line predictions (eliminates ±1-2 variance)
   - Use `location.name` for identification, `location.focus_line` for specific line
3. **False positives possible**: Always review findings
4. **Requires LLM**: Needs vLLM server running
5. **Speed**: Full scan takes ~90 seconds per file
6. **File size limit**: ~1,500 lines max (16K token context)
   - 16K chosen based on analysis of 10M+ GitHub repositories
   - Covers 90-95th percentile of Python files in the wild
   - Median Python file: 258 lines (~2,600 tokens)
   - Mean Python file: 879 lines (~8,800 tokens)
   - With ~500 token prompt overhead, 16K handles up to ~1,500 lines

---

## Key Takeaways for AI Agents

✓ **Use Python API** for efficiency (not subprocess CLI)
✓ **Parse `finding.explanation`** for fix instructions
✓ **Use `finding.location.name`** to find where to fix
✓ **Use `finding.location.snippet`** to identify exact code
✓ **Filter by severity** for faster checks (`--severity critical`)
✓ **Verify fixes** by re-running linter
✓ **Handle errors** gracefully (connection, timeout, OOM)
✓ **Use vLLM utilities** for automated server management (optional)

**Integration Pattern (Manual Server):**
```python
# 1. Write code
write_scientific_code(file_path)

# 2. Check for issues (server running manually)
linter = SciCodeLinter()
result = linter.check_file(Path(file_path))

# 3. Fix issues
for finding in result.findings:
    apply_fix(file_path, finding)

# 4. Verify
assert len(linter.check_file(Path(file_path)).findings) == 0
```

**Integration Pattern (Automated Server):**
```python
from scicode_lint.vllm import VLLMServer

# 1. Write code
write_scientific_code(file_path)

# 2. Auto-start server, check, fix, auto-stop
with VLLMServer():
    linter = SciCodeLinter()
    result = linter.check_file(Path(file_path))

    for finding in result.findings:
        apply_fix(file_path, finding)

    # Verify
    assert len(linter.check_file(Path(file_path)).findings) == 0
```

See [VLLM_UTILITIES.md](VLLM_UTILITIES.md) for automated server management.
