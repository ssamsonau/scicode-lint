# Real-World Scientific ML Code Analysis Report

Analysis of 100 Python files with **ground truth labels** for data leakage issues, from Yang et al. ASE'22 study. scicode-lint findings are compared against human-verified labels to measure detection accuracy.

## Analysis Summary

- **Analysis Date:** 2026-03-16 21:37
- **Report Generated:** 2026-03-16 18:20
- **scicode-lint Version:** 0.2.1
- **Files Analyzed:** 99 / 99
- **Files with Findings:** 11 (11.1%)
- **Total Findings:** 15
- **Ground Truth (aggregate):** Precision 67%, Recall 18%, F1 29%, Accuracy 78%
- **Excluded:** 12 notebooks (pattern timeouts)

## Ground Truth Comparison

Comparison against Yang et al. ASE'22 manual labels (notebooks with pattern timeouts are excluded).
 Excluded: 12 notebooks due to timeouts.

| Label | TP | FP | FN | TN | Precision | Recall | F1 |
|-------|---:|---:|---:|---:|----------:|-------:|---:|
| pre | 6 | 2 | 5 | 25 | 75.0% | 54.5% | 63.2% |
| overlap | 0 | 1 | 2 | 44 | 0.0% | 0.0% | 0.0% |
| multi | 0 | 0 | 20 | 30 | 0.0% | 0.0% | 0.0% |

| Label | Correct | Total | Accuracy |
|-------|--------:|------:|---------:|
| pre | 31 | 38 | 81.6% |
| overlap | 44 | 47 | 93.6% |
| multi | 30 | 50 | 60.0% |

## Verification Summary

**15 findings pending verification.**

## Findings by Category

| Category | Count | Unique Files |
|----------|-------|--------------|
| ai-training | 15 | 11 |

## Findings by Severity

| Severity | Count | % of Total |
|----------|-------|------------|
| Critical | 15 | 100.0% |

## Most Common Patterns

| Pattern | Category | Count | Files |
|---------|----------|-------|-------|
| ml-007 | ai-training | 6 | 6 |
| ml-001 | ai-training | 6 | 6 |
| ml-009 | ai-training | 2 | 2 |
| ml-010 | ai-training | 1 | 1 |

## Example Findings

Representative findings from each category (with links to source):

### ai-training

**ml-007** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-23-nb_2646.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-23-nb_2646.ipynb#L46-L52) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Location:** class `LabelEncoder` (line 49)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-007: Issue detected
- **Explanation:** Data leakage: fit_transform on test data means the test set uses its own statistics instead of training statistics. Use transform() on test data.


- **Suggestion:** Review the code and fix according to the explanation.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ["class","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","thyroid_surgery","query_hypothyroid","query_hyperthyroid","pregnant","sick","tumor","lithium","goitre","TSH_measured","T3_measured","TT4_measured","T4U_measured","FTI_measured","TBG_measured"]
for i in cols:
    data[i] = data[i].astype(str)
    data[i] = le.fit_transform(data[i])
```

**ml-001** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-12-nb_677.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-12-nb_677.ipynb#L19-L25) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Location:** module `<module>` (line 22)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-001: Issue detected
- **Explanation:** Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.


- **Suggestion:** Review the code and fix according to the explanation.

```python
df.columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
```

**ml-001** (critical, 95% confidence)

- **File:** [sample_notebooks/2021-09-05-nb_1633.ipynb](https://github.com/malusamayo/GitHubAPI-Crawler/blob/main/sample_notebooks/2021-09-05-nb_1633.ipynb#L113-L119) in [leakage_paper_sample](https://github.com/malusamayo/GitHubAPI-Crawler)
- **Location:** function `scale_test_data` (line 116)
- **Paper:** Data Leakage in Notebooks: Static Detection and Better Processes
- **Issue:** ml-001: Issue detected
- **Explanation:** Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.


- **Suggestion:** Review the code and fix according to the explanation.

```python
if col == "new_cases":
            df[["new_cases"]] = scaler.transform(df[["new_cases"]])
        if col == "people_fully_vaccinated_per_hundred":
            df[["people_fully_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_fully_vaccinated_per_hundred"]])
        if col == "people_vaccinated_per_hundred":
            df[["people_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_vaccinated_per_hundred"]])
```

---

*Analysis conducted: 2026-03-16 | Report generated: 2026-03-16 18:20*
