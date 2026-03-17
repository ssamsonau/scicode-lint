# Integration Test Results (LLM-as-Judge)

## Overall Summary

- **Scenarios**: 4/4 passed
- **Recall**: 87.9% (29/33 expected bugs)
- **Precision**: 100.0%
- **TP-bonus** (real bugs not in expected): 2
- **False Positives**: 0

## Metrics Breakdown

| Metric | Value |
|--------|-------|
| Expected bugs detected | 29/33 |
| Partial detections | 0 |
| Missed bugs | 4 |
| Bonus true positives | 2 |
| False positives | 0 |

## Scenario Results

### ml_pipeline_complete - ✓ PASS

- Recall: 88.9%
- Correct: 8/9
- Partial: 0/9
- Missed: 1/9
- TP-bonus: 0
- False positives: 0

### pytorch_training_issues - ✓ PASS

- Recall: 83.3%
- Correct: 5/6
- Partial: 0/6
- Missed: 1/6
- TP-bonus: 0
- False positives: 0

### data_preprocessing_bugs - ✓ PASS

- Recall: 81.8%
- Correct: 9/11
- Partial: 0/11
- Missed: 2/11
- TP-bonus: 0
- False positives: 0

### repeated_bugs - ✓ PASS

- Recall: 100.0%
- Correct: 7/7
- Partial: 0/7
- Missed: 0/7
- TP-bonus: 2
- False positives: 0

**Bonus detections (real bugs):**

- `rep-003`: The code contains hardcoded hyperparameters (batch_size=32 and Adam's default learning rate) which are not tracked or configurable. These values are buried in the code and cannot be easily adjusted or reproduced, violating best practices for ML code quality.
- `pt-010`: The DataLoader is created without num_workers, forcing single-process data loading which can create a GPU idle bottleneck. Setting num_workers to 2-8 would enable multi-process data loading with prefetching, improving training throughput as recommended by the linter.
