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

### User Experience

**Users still get complete information:**
- Pattern is detected
- All problematic line numbers listed
- Can navigate to each instance
- Nothing is hidden

### Test Implications

Integration tests account for this:
```yaml
# scenario with 3 training functions missing zero_grad
expected_patterns:
  pt-004: 1  # Expect 1 finding (NOT 3)
  # The single finding contains lines: [54, 70, 87]
```

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

### Decision

**Status**: Accepted as current design

**Rationale**:
- Provides all necessary information
- Simpler to implement and maintain
- Sufficient for current use cases
- Can enhance later if needed
