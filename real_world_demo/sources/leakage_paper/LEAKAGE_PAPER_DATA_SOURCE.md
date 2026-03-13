# Leakage Paper Data Source

Data source from Yang et al. ASE'22 paper: "Data Leakage in Notebooks: Static Detection and Better Processes"

**Paper:** https://arxiv.org/abs/2209.03345
**Tool Repository:** https://github.com/malusamayo/leakage-analysis
**Data Repository:** https://github.com/malusamayo/GitHubAPI-Crawler

---

## What the Paper Detects

The paper's tool uses **static data-flow analysis** with:
1. AST parsing
2. Code transformation to SSA form
3. Type inference (customized Pyright)
4. Datalog fact generation
5. Datalog analysis (Souffle engine)

This is fundamentally different from our LLM-based pattern matching. The paper tracks **data provenance** through code execution paths.

---

## The Three Leakage Types

### 1. Preprocessing Leakage (`pre`)

**Paper's definition:** Training data includes reduced/transformed information from test/validation data.

**Example:**
```python
# LEAKAGE: Scaler learns statistics from ALL data including test
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on full dataset
X_train, X_test = train_test_split(X_scaled, ...)
```

**Detection method:** Datalog tracks when preprocessing operations (fit, fit_transform) receive data that later flows to both train and test sets.

**scicode-lint patterns:**
- `ml-001-scaler-leakage` - Scaler fit before split
- `ml-007-test-set-preprocessing` - General preprocessing before split

**Match quality:** Good conceptual match. Both detect preprocessing applied to combined data.

---

### 2. Overlap Leakage (`overlap`)

**Paper's definition:** Training and test data **derive from the same source** through data-flow analysis.

**Datalog rule:**
```prolog
DataOverlap(train, ctx1, test, ctx2) :-
    FlowFromExtended(test, ctx2, src_equiv, ctx_src, "data"),
    FlowFromExtended(train, ctx1, src_equiv, ctx_src, "equiv").
```

This means: overlap exists when test and train data **flow from a common source variable** with equivalent content.

**Example of what paper detects:**
```python
data = load_data()
# Both derive from same source without verified disjoint split
train_data = data.sample(frac=0.8)
test_data = data.sample(frac=0.2)   # May overlap!
```

**What our ml-009 pattern looks for:**
```python
# Explicit index overlap
train = data[:800]
test = data[700:]  # Indices 700-800 in both

# SMOTE before split
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_resampled, ...)
```

**Updated ml-009 (v2.0.0)** detects:
- Unsafe `.sample()` splits from same source (may overlap)
- Manual slicing with potential overlap
- Both train/test derived from same DataFrame without disjoint split

| Aspect | Paper's Datalog Approach | Our LLM Approach |
|--------|--------------------------|------------------|
| Method | Static data-flow analysis | Pattern recognition in code |
| Detects | Any shared data provenance | Visible unsafe split patterns |
| Precision | High (formal analysis) | Depends on code clarity |

---

### 3. Multi-test Leakage (`multi`)

**Paper's definition:** Only validation data is detected (used for tuning), no separate held-out test set.

**Example:**
```python
# LEAKAGE: "test" set used for hyperparameter tuning
for params in param_grid:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Tuning on "test"
    if score > best_score:
        best_params = params
# No separate held-out test set for final evaluation
```

**Updated ml-010 (v2.0.0)** detects:
- Only train/val split exists, no test set
- CV score reported as final metric (no holdout)
- GridSearchCV best_score_ used as final result

**Match quality:** Direct match with paper's definition.

---

## Ground Truth Labels

File: `data/leakage_paper/ground_truth.csv` (downloaded from paper's repo)

**Columns:**
- `nb` - Notebook path (e.g., `nb_1244.py`)
- `model` - Has ML model (Y/N) - notebooks without models can't have leakage
- `pre` - Preprocessing leakage (Y/N)
- `overlap` - Overlap leakage (Y/N)
- `multi` - Multi-test leakage (Y/N)

**Label format:**
- `Y` or `Y (details)` = positive
- `N` or `N [annotation]` = negative

---

## Pattern Mapping

| Paper Label | scicode-lint Patterns | Description |
|-------------|----------------------|-------------|
| `pre` | ml-001, ml-007 | Preprocessing leakage - direct match |
| `overlap` | ml-009 | Train/test from same source without disjoint split |
| `multi` | ml-010 | No held-out test set, only validation |

All patterns updated (v2.0.0) to match paper's definitions for direct comparison.

### Detection Approach Differences

The paper uses **datalog-based static analysis**:
1. Parse code to AST
2. Convert to SSA form
3. Generate datalog facts about data flow
4. Run transitive closure to find shared origins

Our LLM pattern matching can only detect **visible code patterns** like:
- Overlapping slice indices (`data[:800]` and `data[700:]`)
- SMOTE/oversampling before split
- Explicit duplicate concatenation

We **cannot detect** what the paper catches:
- Variables that flow from same source through arbitrary code paths
- Implicit sharing via object references
- Complex data dependencies across cells

---

## Usage

**IMPORTANT:** Only run the 4 leakage-related patterns for ground truth comparison. Running all 66 patterns is unnecessary and slow.

```bash
# Download and prepare
python -m real_world_demo.sources.leakage_paper --run

# Run analysis (REQUIRED: only leakage patterns for ground truth comparison)
python -m real_world_demo.run_analysis \
    --manifest real_world_demo/data/leakage_paper/manifest.csv \
    --base-dir real_world_demo/collected_code/leakage_paper \
    --patterns ml-001,ml-007,ml-009,ml-010 \
    --max-concurrent 50

# Compare with ground truth
python -m real_world_demo.sources.leakage_paper.compare_ground_truth
python -m real_world_demo.sources.leakage_paper.compare_ground_truth --detailed
```

### Required Patterns

| Pattern | Maps to Paper Label | Description |
|---------|---------------------|-------------|
| `ml-001` | `pre` | Scaler fit before split |
| `ml-007` | `pre` | Preprocessing before split |
| `ml-009` | `overlap` | Train/test from same source |
| `ml-010` | `multi` | No held-out test set |

### Analysis Settings

| Setting | Recommended | Notes |
|---------|-------------|-------|
| `--patterns` | `ml-001,ml-007,ml-009,ml-010` | **Required** - only these 4 patterns map to paper labels |
| `--max-concurrent` | 50 | Higher values (150) cause vLLM overload and timeouts |
| `--timeout` | 120 | Scales automatically based on file size |

**Timeout scaling:** The linter automatically increases timeout for larger files:
- ≤200 lines: base timeout (120s)
- >200 lines: +30s per 200 additional lines
- Maximum: 5× base (600s)
