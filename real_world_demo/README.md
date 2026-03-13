# Real-World Demo: Scientific ML Code Analysis

Collect Python files from scientific ML papers and run scicode-lint to demonstrate findings on real-world research code.

**Goal:** Produce impact statistics like "X% of genomics ML papers had data leakage risks."

## Structure

```
real_world_demo/
├── sources/                         # Data source implementations
│   ├── papers_with_code/            # PapersWithCode repos
│   │   ├── __main__.py              # Entry point
│   │   ├── filter_papers.py         # Filter by domain + exclude ML venues
│   │   ├── filter_abstracts.py      # LLM semantic filter (AI+science?)
│   │   ├── clone_repos.py           # Clone GitHub repos
│   │   ├── filter_files.py          # Find ML files
│   │   ├── prefilter_files.py       # LLM pre-filter (pipeline code?)
│   │   ├── generate_manifest.py     # Create manifest
│   │   └── run_pipeline.py          # Orchestrator
│   └── leakage_paper/               # Yang et al. ASE'22 notebooks
│       ├── __main__.py              # Download/prepare
│       ├── compare_ground_truth.py  # Precision/recall vs labels
│       └── LEAKAGE_PAPER_DATA_SOURCE.md
│
├── run_analysis.py                  # Generic analysis (any source)
├── verify_findings.py               # Claude verification
├── generate_report.py               # Report from database
├── database.py                      # SQLite storage
├── config.py                        # Shared configuration
└── models.py, utils.py              # Shared utilities
```

## Data Sources

| Source | Description | Files | Command |
|--------|-------------|-------|---------|
| **PapersWithCode** | Scientific ML repos from PWC archive | ~1500 .py files | `python -m real_world_demo.sources.papers_with_code --run --papers 100` |
| **Leakage Paper** | Sample notebooks from Yang et al. ASE'22 | 99 .ipynb files | `python -m real_world_demo.sources.leakage_paper --run` |

### Leakage Paper Source

Notebooks from ["Data Leakage in Notebooks"](https://arxiv.org/abs/2209.03345) (Yang et al., ASE'22). Includes ground truth labels.

**See:** [sources/leakage_paper/LEAKAGE_PAPER_DATA_SOURCE.md](sources/leakage_paper/LEAKAGE_PAPER_DATA_SOURCE.md)

| Paper Label | Description | scicode-lint Pattern | F1 Score |
|-------------|-------------|---------------------|----------|
| `pre` | Preprocessing leakage | `ml-001`, `ml-007` | 81.0% |
| `overlap` | Train/test overlap | `ml-009` | 28.6% |
| `multi` | Multi-test leakage | `ml-010` | 39.1% |

**Required patterns for ground truth comparison:** `ml-001,ml-007,ml-009,ml-010`

```bash
# Download and prepare
python -m real_world_demo.sources.leakage_paper --run

# Run analysis (--source sets manifest, base-dir, and patterns automatically)
python -m real_world_demo.run_analysis --source leakage_paper

# Compare with ground truth
python -m real_world_demo.sources.leakage_paper.compare_ground_truth
python -m real_world_demo.sources.leakage_paper.compare_ground_truth --detailed
```

### PapersWithCode Source

Scientific ML repositories filtered by domain (biology, chemistry, medical, etc.).

**See:** [sources/papers_with_code/PAPERS_WITH_CODE_DATA_SOURCE.md](sources/papers_with_code/PAPERS_WITH_CODE_DATA_SOURCE.md)

```bash
# Download, clone, and prepare
python -m real_world_demo.sources.papers_with_code --run --papers 50

# Or use the pipeline orchestrator
python -m real_world_demo.sources.papers_with_code.run_pipeline --papers 50

# Run analysis (--source sets manifest and base-dir automatically)
python -m real_world_demo.run_analysis --source papers_with_code
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[real-world-demo]"

# Option 1: Leakage paper notebooks (faster, no cloning)
python -m real_world_demo.sources.leakage_paper --run
python -m real_world_demo.run_analysis --source leakage_paper

# Option 2: PapersWithCode repos (full pipeline)
python -m real_world_demo.sources.papers_with_code --run --papers 50
python -m real_world_demo.run_analysis --source papers_with_code
```

## Post-Analysis Reporting

Different data sources have different reporting workflows:

### Leakage Paper (Ground Truth Evaluation)

```bash
# 1. Check initial results
python -m real_world_demo.sources.leakage_paper.compare_ground_truth

# 2. Retry timed-out patterns with larger timeout
python -m real_world_demo.run_analysis --retry-timeouts <RUN_ID> --source leakage_paper --timeout 300

# 3. Final comparison after retries
python -m real_world_demo.sources.leakage_paper.compare_ground_truth

# 4. (Optional) Detailed per-notebook breakdown
python -m real_world_demo.sources.leakage_paper.compare_ground_truth --detailed
```

**Output:** Precision/recall/F1 metrics against paper's ground truth labels.

### PapersWithCode (Verification Pipeline)

```bash
# 1. Generate findings report
python -m real_world_demo.generate_report --run-id <RUN_ID>

# 2. Verify findings with Claude (measures precision)
python -m real_world_demo.verify_findings --run-id <RUN_ID>

# 3. Generate verified-only report
python -m real_world_demo.generate_report --run-id <RUN_ID> --verified-only
```

**Output:** Findings report + Claude-verified precision metrics. See [Verification with Claude Opus](#verification-with-claude-opus) for details.

## Pipeline Stages (PapersWithCode)

The pipeline ensures we select papers that are **AI applied to science** (not ML methodology research):

| Stage | Module | Description | Output |
|-------|--------|-------------|--------|
| filter | `filter_papers.py` | Filter by scientific domain keywords, exclude ML venues (NeurIPS, ICML, etc.) | `data/filtered_papers.json` |
| abstract_filter | `filter_abstracts.py` | **LLM semantic filter**: Is this AI applied to real science? | `data/ai_science_papers.json` |
| clone | `clone_repos.py` | Clone GitHub repos from AI+science papers | `cloned_repos/` |
| files | `filter_files.py` | Find Python files with ML imports | `data/qualifying_files.json` |
| prefilter | `prefilter_files.py` | LLM pre-filter: Is this ML pipeline code? | `data/pipeline_files.json` |
| manifest | `generate_manifest.py` | Collect files and create manifest | `collected_code/manifest.csv` |
| analyze | `run_analysis.py` | Run scicode-lint on all files | `data/analysis.db` |
| report | `generate_report.py` | Generate findings report | `reports/findings_*.md` |
| verify | `verify_findings.py` | Verify findings with Claude | Updates database |

### Paper Selection Flow

```
PapersWithCode Archive (500K+ papers)
    ↓
[1] Domain keyword filter (biology, chemistry, medical, etc.)
    ↓
[2] Exclude ML venues (NeurIPS, ICML, ICLR, CVPR, etc.)
    ↓
[3] LLM abstract filter: "Is this AI applied to real science?"
    ↓
AI+Science Papers (~papers with repo URLs)
    ↓
Clone → Filter Files → Analyze
```

This ensures we analyze code from papers that genuinely apply AI/ML to scientific problems, not ML methodology papers that happen to use scientific datasets as benchmarks.

## Database

Results are stored in SQLite (`data/analysis.db`) with data source isolation:

```bash
# List runs (shows data source)
python -m real_world_demo.generate_report --list-runs

# Filter by data source
python -m real_world_demo.generate_report --list-runs --data-source leakage_paper

# Query directly
sqlite3 real_world_demo/data/analysis.db \
    "SELECT id, data_source, total_files, total_findings FROM analysis_runs ORDER BY id DESC LIMIT 5"
```

### Pattern-Level Results

Each pattern execution is tracked in `pattern_runs` table with status: `success`, `timeout`, `context_length`, or `api_error`.

```bash
# Count timeouts per pattern
sqlite3 real_world_demo/data/analysis.db "
SELECT pattern_id, status, COUNT(*) as count
FROM pattern_runs pr
JOIN file_analyses fa ON pr.file_analysis_id = fa.id
WHERE fa.run_id = (SELECT MAX(id) FROM analysis_runs)
GROUP BY pattern_id, status"

# Files where specific pattern timed out
sqlite3 real_world_demo/data/analysis.db "
SELECT f.file_path, pr.error_message
FROM pattern_runs pr
JOIN file_analyses fa ON pr.file_analysis_id = fa.id
JOIN files f ON fa.file_id = f.id
WHERE pr.status = 'timeout' AND pr.pattern_id = 'ml-009'"
```

**Note:** Ground truth comparison (`compare_ground_truth.py`) automatically excludes notebooks where relevant patterns timed out, preventing timeouts from inflating false negative counts.

### Retrying Timeouts

Retry timed-out patterns from a previous run with a larger timeout:

```bash
# Retry timeouts from run 44 with 300s timeout
python -m real_world_demo.run_analysis --retry-timeouts 44 --source leakage_paper --timeout 300
```

This updates the existing `pattern_runs` records in place, changing status from `timeout` to `success` (or another error if still failing).

## Output Structure

```
real_world_demo/
├── data/                          # Intermediate data (gitignored)
│   ├── filtered_papers.json       # Papers after domain/venue filter
│   ├── ai_science_papers.json     # Papers after LLM abstract filter
│   ├── abstract_excluded.json     # Papers rejected by abstract filter
│   ├── analysis.db                # SQLite with all results
│   └── leakage_paper/             # Leakage paper data
│       ├── manifest.csv
│       └── ground_truth.csv
├── cloned_repos/                  # Cloned repositories (gitignored)
├── collected_code/                # Collected files (gitignored)
│   ├── manifest.csv               # PapersWithCode manifest
│   └── leakage_paper/             # Leakage paper files
├── output_examples/               # Example reports (git tracked)
│   ├── papers_with_code/          # PapersWithCode reports
│   │   ├── FINDINGS_REPORT.md
│   │   └── VALID_FINDINGS_SAMPLE.md
│   └── leakage_paper/             # Ground truth dataset
│       ├── FINDINGS_REPORT.md
│       └── GROUND_TRUTH_COMPARISON.md
└── reports/                       # Generated reports (gitignored)
    └── FINDINGS_REPORT_YYYY-MM-DD_HHMM.md
```

## Output Examples

**`output_examples/`** contains tracked reference outputs for documentation. **`reports/`** is gitignored for per-run generated reports.

| Directory | Files | Description |
|-----------|-------|-------------|
| `papers_with_code/` | `FINDINGS_REPORT.md`, `VALID_FINDINGS_SAMPLE.md` | PapersWithCode analysis |
| `leakage_paper/` | `FINDINGS_REPORT.md`, `GROUND_TRUTH_COMPARISON.md` | Ground truth dataset (Yang et al. ASE'22) |

### Updating Output Examples

After running analysis and generating reports, copy to `output_examples/` for tracking.

**Generated files** (in `reports/<source>/`, gitignored):
| Script | Generated File |
|--------|----------------|
| `generate_report` | `findings_YYYY-MM-DD_HHMM.md` |
| `generate_report --verified-only` | `valid_findings_YYYY-MM-DD_HHMM.md` |
| `compare_ground_truth` | (stdout) |

**Target files** (in `output_examples/<source>/`, tracked):
| Source | File | From |
|--------|------|------|
| `leakage_paper/` | `GROUND_TRUTH_COMPARISON.md` | `compare_ground_truth` stdout |
| `leakage_paper/` | `FINDINGS_REPORT.md` | `findings_*.md` |
| `papers_with_code/` | `FINDINGS_REPORT.md` | `findings_*.md` |
| `papers_with_code/` | `VALID_FINDINGS_SAMPLE.md` | `valid_findings_*.md` |

```bash
# Leakage paper
python -m real_world_demo.sources.leakage_paper.compare_ground_truth \
    > output_examples/leakage_paper/GROUND_TRUTH_COMPARISON.md
python -m real_world_demo.generate_report --run-id <RUN_ID> --data-source leakage_paper
cp reports/leakage_paper/findings_YYYY-MM-DD_HHMM.md \
    output_examples/leakage_paper/FINDINGS_REPORT.md

# PapersWithCode
python -m real_world_demo.generate_report --run-id <RUN_ID>
cp reports/papers_with_code/findings_YYYY-MM-DD_HHMM.md \
    output_examples/papers_with_code/FINDINGS_REPORT.md
python -m real_world_demo.generate_report --run-id <RUN_ID> --verified-only
cp reports/papers_with_code/valid_findings_YYYY-MM-DD_HHMM.md \
    output_examples/papers_with_code/VALID_FINDINGS_SAMPLE.md
```

**Important:** Review output before copying. These files are git-tracked as reference documentation.

## Scientific Domains (PapersWithCode)

Papers are filtered by task keywords:

- **biology**: protein, gene, cell, dna, rna, genomic, ...
- **chemistry**: molecule, drug, compound, chemical, ...
- **medical**: medical, clinical, disease, diagnosis, ...
- **physics**: physics, quantum, particle, ...
- **materials**: material, crystal, alloy, polymer, ...
- **neuroscience**: brain, neural, fmri, neuroimaging, ...
- **earth_science**: climate, weather, earthquake, satellite, ...
- **astronomy**: galaxy, stellar, exoplanet, supernova, ...

## Benchmark Results

### Leakage Paper (Yang et al. ASE'22)

Comparison against ground truth labels from the paper. See [output_examples/leakage_paper/GROUND_TRUTH_COMPARISON.md](output_examples/leakage_paper/GROUND_TRUTH_COMPARISON.md) for full output.

Run `python -m real_world_demo.sources.leakage_paper.compare_ground_truth` for latest results.

| Label | Patterns | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| `pre` | ml-001, ml-007 | 68.0% | 100.0% | 81.0% |
| `overlap` | ml-009 | 25.0% | 33.3% | 28.6% |
| `multi` | ml-010 | 52.9% | 31.0% | 39.1% |

### PapersWithCode

> **Note:** Detection pending re-run due to a bug that reduced accuracy (~99%→78%).
> Verified findings remain valid (Opus verification unaffected). Precision expected to improve.

**Pipeline:** 45 papers → 61 repos → 884 files collected.

Quick test on 15 files (full analysis pending):

| Metric | Value |
|--------|-------|
| Files with findings | 53.3% (8/15) |
| Total findings | 19 |
| **Verified precision** | **15.8%** (3/19 valid) |

**Valid patterns:**
| Pattern | Issue | Valid | Total | Precision |
|---------|-------|-------|-------|-----------|
| pt-003 | In-place gradient ops | 1 | 1 | 100% |
| rep-002 | CUDA non-determinism | 2 | 5 | 40% |

**Known issues affecting precision:**
- Empty code snippets in some detections (bug)
- Missing cross-file context (seeds set elsewhere)
- Some patterns too broad (rep-003 hardcoded params)

## Verification with Claude Opus

The verification stage uses Claude Opus to evaluate whether detected findings are real issues or false positives. This provides ground truth for measuring detection precision.

### How It Works

1. **Detection** (vLLM/Qwen3 local): Fast bulk scanning → findings stored in `findings` table
2. **Verification** (Claude Opus): Reviews each finding with full code context
3. **Verdict**: `VALID` (real issue), `INVALID` (false positive), or `UNCERTAIN`
4. **Storage**: Verdicts + reasoning → `finding_verifications` table

### Usage

```bash
# Verify all findings from latest run
python -m real_world_demo.verify_findings

# Verify specific run
python -m real_world_demo.verify_findings --run-id 43

# Verify by category
python -m real_world_demo.verify_findings --category data-leakage

# Limit batch size (useful while analysis is still running)
python -m real_world_demo.verify_findings --limit 100

# Resume verification (skip already-verified findings)
python -m real_world_demo.verify_findings --run-id 43 --skip-verified

# Adjust parallelism (default: 3)
python -m real_world_demo.verify_findings --parallel 5
```

### Incremental Verification

You can run verification while analysis is still running. When analysis adds more findings:

```bash
# Resume to verify only new findings
python -m real_world_demo.verify_findings --run-id 43 --skip-verified
```

### Check Results

```bash
# Verification stats
sqlite3 real_world_demo/data/analysis.db "
SELECT status, COUNT(*) as count
FROM finding_verifications
GROUP BY status"

# Precision by pattern
sqlite3 real_world_demo/data/analysis.db "
SELECT fn.pattern_id,
       SUM(CASE WHEN fv.status='valid' THEN 1 ELSE 0 END) as valid,
       COUNT(*) as total,
       ROUND(100.0 * SUM(CASE WHEN fv.status='valid' THEN 1 ELSE 0 END) / COUNT(*), 1) as precision
FROM finding_verifications fv
JOIN findings fn ON fn.id = fv.finding_id
GROUP BY fn.pattern_id
ORDER BY total DESC"
```

## Requirements

- Python 3.13+
- `datasets` library (HuggingFace)
- `git` command available
- scicode-lint installed
- vLLM server running (for analysis stage)
- Claude CLI authenticated (for verify stage): `claude login`
