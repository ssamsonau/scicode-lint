# Contributing to Integration Tests

## Adding New Scenarios

### Important: One Finding Per Pattern

Before adding scenarios, understand this design choice:

**Each pattern returns ONE finding per file**, even if multiple instances exist. The finding's `lines` array contains all problematic line numbers.

Example:
```python
# File with 3 missing zero_grad instances
# Expected: pt-004: 1 (not 3)
# Finding will have lines: [54, 70, 87]
```

See [DESIGN_NOTES.md](DESIGN_NOTES.md) for details.

### Step 1: Write Clean Code

Create a Python file in `scenarios/` with **realistic code containing bugs**.

**CRITICAL: No data leakage!**

❌ **NEVER include:**
- Bug annotations: `# BUG:`, `# FIXME:`, `# TODO:`
- Correctness hints: `# should`, `# wrong`, `# missing`, `# incorrect`
- Pattern references: `# ml-001`, `# data leakage`, `# missing train mode`
- Explanatory comments about what's wrong

✅ **DO include:**
- Module docstring explaining what the code does (purpose)
- Function docstrings explaining logic
- Normal code comments explaining complex operations

### Example Template

```python
"""
<Brief description of what this module does>.

This module implements <functionality> for <purpose>.
"""

import torch
import torch.nn as nn

def train_model(model, data):
    """Train the model on provided data."""
    optimizer = torch.optim.Adam(model.parameters())

    for batch in data:
        outputs = model(batch)
        loss = nn.functional.mse_loss(outputs, batch.labels)
        loss.backward()
        optimizer.step()
```

### Step 2: Document Expected Bugs

Add your scenario to `expected_findings.yaml`:

```yaml
scenarios:
  your_scenario_name:
    description: "Brief description of the scenario"
    file: "scenarios/your_file.py"

    expected_patterns:
      pattern-id-1: 1    # Expected count
      pattern-id-2: 2

    min_total_findings: 3
    max_total_findings: 3
    max_false_positives: 0

    bugs:
      - line: 42
        pattern: pattern-id-1
        description: "What bug is at this line"

      - line: 58
        pattern: pattern-id-2
        description: "What bug is at this line"
```

### Step 3: Test Your Scenario

```bash
# Run your scenario
python evals/integration/run_integration_eval.py --scenario your_scenario_name -v

# Check results
# - All expected patterns detected?
# - No false positives?
# - Line numbers match?
```

### Step 4: Verify No Data Leakage

Before committing, check for leakage:

```bash
# Should return NOTHING
grep -i "bug\|fixme\|todo\|wrong\|missing.*mode\|should.*call" scenarios/your_file.py
```

## Reviewing Scenario PRs

When reviewing PRs that add/modify scenarios, check:

1. **No data leakage in code**
   - No bug annotations or hints
   - Only descriptive docstrings

2. **Expected findings are documented**
   - All bugs listed in `expected_findings.yaml`
   - Line numbers are accurate

3. **Code is realistic**
   - Not contrived or obvious
   - Represents real-world mistakes

4. **Scenario runs successfully**
   - Linter finds expected bugs
   - No false positives
   - No crashes

## Common Mistakes

### ❌ Too Obvious
```python
def train():
    # This is broken - missing model.train()
    pass
```

### ✅ Realistic
```python
def train_model(model, data):
    """Train the model on training data."""
    optimizer = Adam(model.parameters())

    for batch in data:
        outputs = model(batch)
        loss.backward()
        optimizer.step()
```

### ❌ Data Leakage
```python
# BUG: Fitting scaler on test data causes data leakage
scaler.fit(X_test)
```

### ✅ Clean
```python
scaler = StandardScaler()
scaler.fit(X_test)
X_train = scaler.transform(X_train)
```

## Testing Checklist

Before submitting:

- [ ] No bug annotations in scenario file
- [ ] Module docstring describes purpose (not bugs)
- [ ] Expected findings documented in YAML
- [ ] Line numbers in YAML match actual bugs
- [ ] Scenario runs: `python evals/integration/run_integration_eval.py --scenario NAME -v`
- [ ] All expected bugs detected
- [ ] Zero false positives
- [ ] grep check: `grep -i "bug\|fixme\|todo" scenarios/your_file.py` returns nothing

## Questions?

- See `README.md` for detailed documentation
- Check existing scenarios for examples
- Ask in PR if unsure about data leakage
