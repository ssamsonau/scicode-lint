# Integration Test Results

## Overall Summary

- **Scenarios**: 0/1 passed
- **Recall**: 100.0% (2/2 expected bugs detected)
- **Precision**: 28.6%
- **False Positives**: 5

## Threshold Checks

- Recall = 100%: ✓ PASS (100.0%)
- False positives = 0: ✗ FAIL (5)

## Scenario Results

### gen_20260315_231832_PyTorch_training_loop_with_gra - ✗ FAIL

- Expected: 2 bugs
- Found: 7 bugs
- False positives: 5

**Pattern breakdown:**

- ✓ `pt-009`: 1/1
- ✓ `pt-012`: 1/1

**False positives:**

- rep-002 (1 findings)
- rep-003 (1 findings)
- rep-004 (1 findings)
- pt-010 (1 findings)
- pt-004 (1 findings)
