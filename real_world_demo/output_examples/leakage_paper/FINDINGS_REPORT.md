# Real-World Scientific ML Code Analysis Report

Analysis of 100 Python files with **ground truth labels** for data leakage issues, from Yang et al. ASE'22 study. scicode-lint findings are compared against human-verified labels to measure detection accuracy.

## Analysis Summary

- **Analysis Date:** 2026-03-13 14:27
- **Report Generated:** 2026-03-13 10:56
- **scicode-lint Version:** 0.1.6
- **Files Analyzed:** 99 / 99
- **Files with Findings:** 64 (64.6%)
- **Total Findings:** 93

## Verification Summary

**93 findings pending verification.**

## Findings by Category

| Category | Count | Unique Files |
|----------|-------|--------------|
| ai-training | 93 | 64 |

## Findings by Severity

| Severity | Count | % of Total |
|----------|-------|------------|
| Critical | 93 | 100.0% |

## Most Common Patterns

| Pattern | Category | Count | Files |
|---------|----------|-------|-------|
| ml-010 | ai-training | 39 | 39 |
| ml-001 | ai-training | 25 | 25 |
| ml-007 | ai-training | 17 | 17 |
| ml-009 | ai-training | 12 | 12 |

## Example Findings

Representative findings from each category (with links to source):

### ai-training

**ml-010** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-21-nb_2685.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-21-nb_2685.ipynb) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-010: Issue detected
- **Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.

- **Suggestion:** Review the code and fix according to the explanation.

**ml-010** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-11-nb_605.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-11-nb_605.ipynb) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-010: Issue detected
- **Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.

- **Suggestion:** Review the code and fix according to the explanation.

**ml-009** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-11-nb_605.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-11-nb_605.ipynb) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-009: Issue detected
- **Explanation:** Overlap leakage: Train and test data derive from the same source without guaranteed disjoint splitting. Use sklearn.model_selection.train_test_split() or verify indices are disjoint.

- **Suggestion:** Review the code and fix according to the explanation.

---

*Analysis conducted: 2026-03-13 | Report generated: 2026-03-13 10:56*
