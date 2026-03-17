# Output Examples

Example reports from scicode-lint analysis, organized by data source.

## About This Directory

These are **committed copies** of reports generated in `real_world_demo/reports/` (gitignored). When reports are updated, copy the latest versions here.

## Structure

```
output_examples/
├── papers_with_code/          # PapersWithCode scientific papers
│   ├── FINDINGS_REPORT.md     # Main findings report (verified)
│   └── VALID_FINDINGS_SAMPLE.md  # 10 valid findings (quick verification)
└── leakage_paper/             # Yang et al. ASE'22 ground truth
    └── FINDINGS_REPORT.md     # Findings vs ground truth labels
```

## Data Sources

### PapersWithCode (`papers_with_code/`)

Scientific ML code from the meta loop set (38 papers, 120 self-contained files). See [`paper_sets/`](../paper_sets/) for dataset details.

| File | Description |
|------|-------------|
| `FINDINGS_REPORT.md` | Full analysis with verified findings |
| `VALID_FINDINGS_SAMPLE.md` | 10 valid findings (critical + high) with pattern diversity |

### Leakage Paper (`leakage_paper/`)

Yang et al. ASE'22 dataset with ground truth labels for data leakage. Used for precision/recall benchmarking.

| File | Description |
|------|-------------|
| `FINDINGS_REPORT.md` | Findings compared against human-verified labels |

## Updating Examples

```bash
# Generate reports
python -m real_world_demo.generate_report --run-id <ID> --verified-only

# Copy to output_examples
cp reports/FINDINGS_REPORT_<date>.md output_examples/papers_with_code/FINDINGS_REPORT.md
cp reports/VALID_FINDINGS_SAMPLE_<date>.md output_examples/papers_with_code/VALID_FINDINGS_SAMPLE.md
```
