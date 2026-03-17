# Integration Test Results

## Overall Summary

- **Scenarios**: 0/4 passed
- **Recall**: 86.5% (32/37 expected bugs detected)
- **Precision**: 80.0%
- **False Positives**: 8

## Threshold Checks

- Recall = 100%: ✗ FAIL (86.5%)
- False positives = 0: ✗ FAIL (8)

## Scenario Results

### ml_pipeline_complete - ✗ FAIL

- Expected: 11 bugs
- Found: 14 bugs
- False positives: 4

**Pattern breakdown:**

- ✓ `ml-001`: 1/1
- ✓ `ml-002`: 1/1
- ✓ `ml-007`: 1/1
- ✓ `pt-004`: 1/1
- ✓ `rep-001`: 1/1
- ✓ `rep-003`: 1/1
- ✓ `rep-004`: 1/1
- ✓ `rep-005`: 1/1
- ✓ `rep-008`: 1/1
- ✓ `rep-009`: 1/1
- ✗ `num-005`: 0/1

**Missing patterns:**

- num-005 (expected 1, found 0)

**False positives:**

- pt-011 (1 findings)
- ml-005 (1 findings)
- ml-010 (1 findings)
- ml-008 (1 findings)

### pytorch_training_issues - ✗ FAIL

- Expected: 8 bugs
- Found: 8 bugs
- False positives: 1

**Pattern breakdown:**

- ✓ `pt-001`: 1/1
- ✓ `pt-004`: 1/1
- ✓ `pt-007`: 1/1
- ✓ `pt-010`: 1/1
- ✓ `pt-011`: 1/1
- ✓ `pt-015`: 1/1
- ✓ `pt-021`: 1/1
- ✗ `rep-002`: 0/1

**Missing patterns:**

- rep-002 (expected 1, found 0)

**False positives:**

- ml-008 (1 findings)

### data_preprocessing_bugs - ✗ FAIL

- Expected: 11 bugs
- Found: 9 bugs
- False positives: 0

**Pattern breakdown:**

- ✓ `ml-001`: 1/1
- ✓ `ml-002`: 1/1
- ✗ `ml-003`: 0/1
- ✓ `ml-005`: 1/1
- ✓ `ml-006`: 1/1
- ✓ `ml-007`: 1/1
- ✓ `rep-001`: 1/1
- ✓ `rep-003`: 1/1
- ✓ `rep-004`: 1/1
- ✗ `num-005`: 0/1
- ✓ `py-002`: 1/1

**Missing patterns:**

- ml-003 (expected 1, found 0)
- num-005 (expected 1, found 0)

### repeated_bugs - ✗ FAIL

- Expected: 7 bugs
- Found: 9 bugs
- False positives: 3

**Pattern breakdown:**

- ✓ `ml-001`: 1/1
- ✓ `ml-007`: 1/1
- ✓ `ml-008`: 1/1
- ✓ `pt-004`: 1/1
- ✓ `rep-001`: 1/1
- ✓ `rep-004`: 1/1
- ✗ `num-005`: 0/1

**Missing patterns:**

- num-005 (expected 1, found 0)

**False positives:**

- rep-002 (1 findings)
- rep-003 (1 findings)
- pt-010 (1 findings)
