# Integration Tests - Design Notes

## One Finding Per Pattern Per File

### Current Behavior

Each pattern check returns **exactly ONE finding per file**, regardless of how many instances of the bug exist.

When multiple instances are found, they are **consolidated into a single finding** with all line numbers listed:

```python
# File has 3 functions all missing optimizer.zero_grad()
finding = Finding(
    id="pt-004",
    location=Location(
        lines=[54, 70, 87],  # All three instances
        snippet="..."
    ),
    explanation="Missing optimizer.zero_grad()",
    confidence=0.95
)
```

### Architecture Reasoning

**Why one finding per pattern?**

1. **Detection Model**: Each pattern asks one yes/no question about the entire file
   - "Does this file have training loops missing zero_grad?"
   - Answer: "Yes, on lines 54, 70, 87"

2. **LLM Response**: Structured output returns single DetectionResult per pattern
   ```python
   class DetectionResult(BaseModel):
       detected: Literal["yes", "no", "context-dependent"]
       lines: list[int]      # All instances in one list
       confidence: float
       reasoning: str
   ```

3. **Implementation**: `_create_findings()` always returns `[finding]` (list with one item)
   ```python
   def _create_findings(...) -> list[Finding]:
       finding = Finding(...)
       return [finding]  # Single finding
   ```

### User Experience

**Users still get complete information:**
- ✅ Pattern is detected
- ✅ All problematic line numbers listed
- ✅ Can navigate to each instance
- ✅ Nothing is hidden

**Example output:**
```
test.py — 1 issue found

🔴 CRITICAL [lines 54, 70, 87] pt-004: Issue detected
   Missing optimizer.zero_grad() in training loop causes
   gradient accumulation bugs.

   Found in 3 locations:
   - Line 54: train_epoch_v1()
   - Line 70: train_epoch_v2()
   - Line 87: train_epoch_v3()
```

### Test Implications

Integration tests must account for this behavior:

```yaml
# repeated_bugs.py has 3 training functions missing zero_grad
expected_patterns:
  pt-004: 1  # Expect 1 finding (NOT 3)

# The single finding should contain lines: [54, 70, 87]
```

### Comparison to Other Linters

**ESLint / Pylint behavior:**
- Return multiple separate findings for multiple instances
- Each violation = separate finding

**Our behavior:**
- Return one finding per pattern with all instances
- Grouped by pattern type

### Trade-offs

**Advantages:**
- Cleaner data model (one finding per pattern per file)
- Easier to understand: "This file has the pt-004 problem"
- All instances listed together
- Simpler implementation

**Disadvantages:**
- Can't track "finding fixed" at instance level
- Different from traditional linters
- Might need enhancement for very large files (50+ instances)

### Future Enhancement Options

If needed, could be extended to:

1. **Split threshold**: Return separate findings if > N instances
   ```python
   if len(lines) > 10:
       return [Finding(...) for line_group in split_lines()]
   ```

2. **Per-function findings**: Detect instance boundaries and split
   ```python
   # Detect that lines 54, 70, 87 are in different functions
   # Return 3 findings (one per function)
   ```

3. **User preference**: Config option for behavior
   ```python
   config = LinterConfig(multi_instance_mode="separate")  # or "consolidated"
   ```

### Decision

**Status**: Accepted as current design ✅

**Rationale**:
- Provides all necessary information
- Simpler to implement and maintain
- Sufficient for current use cases
- Can enhance later if needed

**Documented in**:
- evals/integration/README.md
- evals/integration/expected_findings.yaml
- This file (DESIGN_NOTES.md)
