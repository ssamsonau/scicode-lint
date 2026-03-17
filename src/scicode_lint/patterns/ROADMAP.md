# Pattern Coverage Roadmap

Strategic guide for pattern development in scicode-lint.

---

## Table of Contents

- [Target Audience](#target-audience)
- [Philosophy: Why These Categories?](#philosophy-why-these-categories)
- [Current Coverage (Phase 1)](#current-coverage-phase-1)
- [Library Ecosystem Analysis](#library-ecosystem-analysis)
- [Future Priorities](#future-priorities)
- [Specific Pattern Ideas](#specific-pattern-ideas)
- [Non-Goals](#non-goals)

---

## Target Audience

**scicode-lint targets scientists and researchers writing AI/ML code** - not software engineers building production ML systems.

This distinction matters for prioritization:

| Scientist/Researcher | Production ML Engineer |
|---------------------|----------------------|
| PyTorch (75-80% of papers) | TensorFlow (production deployment) |
| Jupyter notebooks | Kubernetes, Docker |
| Experiment correctness | System reliability |
| Reproducibility for papers | A/B testing frameworks |
| Single-GPU or small cluster | Large-scale distributed |

**Research shows:**
- [75-80% of new ML research papers use PyTorch](https://arxiv.org/html/2508.04035v1) (NeurIPS 2023)
- TensorFlow still dominates production but is [declining in research](https://www.infoworld.com/article/2336447/tensorflow-pytorch-and-jax-choosing-a-deep-learning-framework.html)
- XGBoost is [heavily cited in scientific papers](https://dl.acm.org/doi/10.1145/2939672.2939785) across biology, chemistry, physics, and medicine
- JAX is growing in ML research, especially at Google/DeepMind labs

---

## Philosophy: Why These Categories?

scicode-lint focuses on **bugs that produce silently wrong results** in scientific/ML code. These are the worst kind of bugs: code runs successfully, but conclusions are invalid.

### Category Rationale

| Category | Why It Exists |
|----------|---------------|
| **ai-training** | Training bugs cause models to learn incorrectly. Data leakage inflates metrics, making researchers believe their model works better than it does. Missing gradients prevent learning entirely. These bugs waste compute and lead to wrong conclusions. |
| **ai-inference** | Inference bugs cause trained models to behave differently than expected. Missing `eval()` mode means dropout is still active. Device mismatches cause silent failures or wrong results. |
| **scientific-numerical** | Floating-point arithmetic is counterintuitive. `0.1 + 0.2 != 0.3`. Catastrophic cancellation loses precision. These bugs produce plausible-looking but wrong numbers. |
| **scientific-performance** | Performance bugs don't produce wrong results, but they make scientific iteration painfully slow. Python loops over NumPy arrays are 100x slower than vectorized code. |
| **scientific-reproducibility** | Irreproducible results undermine science. Random seeds, non-deterministic operations, and platform dependencies mean "same code, different results." |

### What We Don't Cover

- **Style issues** - Use ruff, black, pylint
- **Type errors** - Use mypy, pyright
- **Security vulnerabilities** - Use bandit, semgrep
- **General bugs** - Use pytest, hypothesis
- **API misuse without scientific impact** - Focus on correctness, not best practices

---

## Current Coverage (Phase 1)

**Phase 1 focus: PyTorch + scikit-learn + core scientific computing**

This was a deliberate choice: go deep on the dominant research framework before expanding.

**Total: 66 patterns** across 5 categories

| Category | Count | Focus Areas |
|----------|-------|-------------|
| ai-training | 19 | Data leakage, gradient flow, loss functions, PyTorch modes |
| ai-inference | 12 | Eval mode, no_grad, device handling, JIT/ONNX export |
| scientific-numerical | 10 | Float precision, array mutations, overflow/underflow |
| scientific-performance | 11 | Vectorization, threading, memory allocation |
| scientific-reproducibility | 14 | Random seeds, CUDA determinism, file ordering |

### Library Breakdown

| Library | Patterns | % of Total | Status |
|---------|----------|------------|--------|
| PyTorch | 21 | 32% | Good coverage |
| scikit-learn | 10 | 15% | Data leakage focus |
| NumPy | 6 | 9% | Basic coverage |
| CUDA/GPU | 6 | 9% | Determinism focus |
| pandas | 2 | 3% | Minimal |
| Framework-agnostic | 21 | 32% | Reproducibility, numerical |

---

## Library Ecosystem Analysis

Based on research into scientific Python usage patterns.

### Deep Learning Frameworks

| Framework | Research Usage | Production Usage | Priority for scicode-lint |
|-----------|---------------|------------------|---------------------------|
| **PyTorch** | 75-80% of papers | Growing | **HIGH** - current focus |
| **JAX** | Growing (Google/DeepMind) | Niche | **HIGH** - research trend |
| **TensorFlow** | Declining | Still dominant | **MEDIUM** - less urgent for researchers |

### Gradient Boosting (Tabular ML)

XGBoost and related libraries are [heavily used in scientific research](https://pmc.ncbi.nlm.nih.gov/articles/PMC9647306/):

| Domain | Example Applications |
|--------|---------------------|
| **Biology** | [Protein-protein interaction prediction](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.752732/full), gene expression |
| **Medicine** | [Myocardial infarction prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC9647306/), diagnosis models |
| **Drug Discovery** | [QSAR models, clinical trial analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11895769/) |
| **Physics** | Solar cell optimization, materials science |
| **Environmental** | [Harmful algal bloom forecasting](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611362/) |

| Library | Research Citations | Priority |
|---------|-------------------|----------|
| **XGBoost** | Most cited | **HIGH** |
| **LightGBM** | Second, faster training | **MEDIUM** |
| **CatBoost** | Third, categorical handling | **LOW** |

### NLP/LLM

| Library | Usage | Priority |
|---------|-------|----------|
| **Transformers (HF)** | De facto standard for NLP research | **HIGH** |
| **spaCy** | More production-focused | **LOW** |

### Domain-Specific Scientific Libraries

These are used alongside ML frameworks, but bugs are typically domain-specific rather than ML-generic:

| Domain | Libraries | Priority |
|--------|-----------|----------|
| Astronomy | Astropy | Low - domain-specific |
| Bioinformatics | Biopython | Low - domain-specific |
| Physics | SymPy, SciPy | Medium - numerical bugs |
| Neuroscience | Nilearn, MNE | Low - domain-specific |

### Core Data Stack

| Library | Scientific Usage | Priority |
|---------|-----------------|----------|
| **NumPy** | Foundation of everything | **HIGH** - expand coverage |
| **pandas** | Data loading/preprocessing | **MEDIUM** - common bugs |
| **SciPy** | Optimization, stats, signal processing | **MEDIUM** |

---

## Future Priorities

### Priority 1: High-Impact for Scientific Research

#### XGBoost/LightGBM (Target: 10+ patterns)

**Why HIGH priority:** Dominant in scientific papers across biology, chemistry, physics, medicine. Data leakage patterns are critical - scientists using XGBoost for classification/regression need the same protection as deep learning users.

**Key areas:**
- Data leakage through early stopping validation set
- `eval_set` contamination with test data
- Categorical feature encoding before split (leakage)
- Feature importance computed on test data
- Cross-validation with time series (temporal leakage)
- Missing value handling inconsistencies between train/test
- Hyperparameter tuning leakage (tuning on test set)

#### JAX (Target: 8+ patterns)

**Why HIGH priority:** Growing rapidly in ML research. Functional paradigm creates unique bug patterns not covered by PyTorch patterns.

**Key areas:**
- `jax.jit` recompilation overhead (shapes, dtypes)
- Pure function violations (side effects in jitted functions)
- Random key reuse (same key = same "random" numbers)
- Random key not split properly
- `vmap`/`pmap` shape mismatches
- Tracer leakage (tracers escaping jitted scope)
- Pytree structure mismatches
- Device placement bugs

#### Hugging Face Transformers (Target: 10+ patterns)

**Why HIGH priority:** Standard for NLP/LLM research. Complex API with many silent failure modes.

**Key areas:**
- Tokenizer train/inference mismatch
- Padding side mismatch (left vs right)
- Truncation inconsistencies
- `model.eval()` vs `model.train()` (different defaults than base PyTorch)
- `generate()` configuration bugs (temperature=0 with sampling)
- Gradient checkpointing misuse
- LoRA/PEFT: base model not frozen
- Model loading dtype mismatches (fp16 vs fp32)

### Priority 2: Expand Existing Coverage

#### PyTorch Gaps (Target: 8+ new patterns)

Current coverage is good for single-GPU training. Gaps in modern research workflows:

- **Distributed training (DDP)**
  - Missing `model = DDP(model)` wrapper
  - Gradient sync issues across ranks
  - `find_unused_parameters` misuse
  - SyncBatchNorm not used with DDP

- **Mixed precision (AMP)**
  - `torch.autocast` scope issues
  - GradScaler misuse
  - Loss scaling bugs
  - Ops that don't support float16

- **Checkpointing**
  - Not saving optimizer state
  - Not saving LR scheduler state
  - Not saving random state (reproducibility)

#### NumPy/SciPy Gaps (Target: 8+ patterns)

Core scientific computing has limited coverage:

- **Linear algebra**
  - Singular matrix detection before inverse
  - Numerical instability in eigenvalue computation
  - Condition number warnings

- **Optimization (scipy.optimize)**
  - Gradient not provided for gradient-based methods
  - Bounds violated silently
  - Convergence not checked

- **Statistics**
  - Multiple comparison correction missing
  - p-hacking patterns (many tests, no correction)

#### pandas Gaps (Target: 6+ patterns)

Scientists use pandas for data loading, where bugs affect downstream ML:

- **Merge/join bugs**
  - Duplicate key row multiplication
  - Unexpected NaN introduction

- **Data type bugs**
  - Silent int-to-float conversion on NaN
  - Mixed types in column

- **Temporal bugs**
  - Timezone-naive datetime comparisons
  - Sorting without stable sort for ties

### Priority 3: Lower Priority

#### TensorFlow/Keras (Target: 8+ patterns)

**Why MEDIUM (not HIGH):** Still important but declining in research. Scientists are migrating to PyTorch. Worth covering but not urgent.

**Key areas:**
- `training=True/False` in custom layers
- GradientTape scope issues
- `tf.function` retracing
- Eager vs graph mode mismatches

#### PyTorch Lightning (Target: 4+ patterns)

Popular research framework built on PyTorch:

- Callback ordering issues
- `self.log` in wrong hook
- Distributed sync bugs

---

## Specific Pattern Ideas

### XGBoost Patterns

| ID | Name | Description |
|----|------|-------------|
| xgb-001 | early-stop-leakage | Early stopping `eval_set` overlaps with test data |
| xgb-002 | categorical-before-split | Encoding categoricals before train/test split |
| xgb-003 | feature-importance-test | Computing feature importance on test predictions |
| xgb-004 | cv-temporal-early-stop | Using future data for early stopping in time series |
| xgb-005 | eval-metric-mismatch | Training metric differs from evaluation metric |
| xgb-006 | missing-value-train-test | Different missing value handling train vs test |
| xgb-007 | hyperopt-test-leakage | Hyperparameter tuning uses test set performance |

### JAX Patterns

| ID | Name | Description |
|----|------|-------------|
| jax-001 | random-key-reuse | Same random key used multiple times |
| jax-002 | key-not-split | Using key without splitting for multiple calls |
| jax-003 | jit-side-effects | Side effects in `jax.jit` decorated function |
| jax-004 | tracer-leak | Tracer object escapes jitted function scope |
| jax-005 | vmap-shape-mismatch | Inconsistent batch dimensions in vmap |
| jax-006 | jit-retrace-shapes | Dynamic shapes causing excessive retracing |
| jax-007 | pytree-structure | Mismatched pytree structures in transforms |

### Transformers Patterns

| ID | Name | Description |
|----|------|-------------|
| hf-001 | tokenizer-mismatch | Different tokenizer settings train vs inference |
| hf-002 | padding-side-mismatch | Padding side differs between training and generation |
| hf-003 | generate-temp-zero | Temperature=0 with `do_sample=True` |
| hf-004 | lora-base-unfrozen | LoRA applied but base model gradients enabled |
| hf-005 | model-dtype-mismatch | Loading model in wrong dtype for hardware |
| hf-006 | truncation-silent | Sequences truncated without warning |
| hf-007 | attention-mask-missing | Missing attention mask with padded inputs |

### NumPy/SciPy Patterns

| ID | Name | Description |
|----|------|-------------|
| np-linalg-001 | singular-matrix | Matrix inversion without checking condition number |
| np-linalg-002 | eigenvalue-instability | Eigenvalue computation on ill-conditioned matrix |
| scipy-opt-001 | no-gradient-bfgs | Using BFGS without providing gradient |
| scipy-opt-002 | convergence-unchecked | Optimization result not checked for convergence |
| scipy-stats-001 | multiple-comparison | Multiple statistical tests without correction |

### PyTorch DDP Patterns

| ID | Name | Description |
|----|------|-------------|
| pt-ddp-001 | missing-ddp-wrap | Training distributed without DDP wrapper |
| pt-ddp-002 | unused-params | `find_unused_parameters` needed but not set |
| pt-ddp-003 | no-sync-batchnorm | BatchNorm without SyncBatchNorm in DDP |
| pt-amp-001 | autocast-scope | Operations outside autocast context |
| pt-amp-002 | gradscaler-missing | Mixed precision without GradScaler |

---

## Non-Goals

Patterns we explicitly won't add:

| Category | Reason |
|----------|--------|
| **Code style** | Covered by ruff, black, isort |
| **Type errors** | Covered by mypy, pyright |
| **Import errors** | Covered by IDE, Python itself |
| **Deprecated APIs** | Better handled by library-specific linters |
| **Version-specific bugs** | Too fragile, changes with library updates |
| **Performance micro-optimizations** | Focus on correctness, not speed |
| **Framework preferences** | No "use PyTorch instead of TensorFlow" |
| **Domain-specific science bugs** | Astropy usage, Biopython sequences - too specialized |

---

## Contributing New Patterns

See [README.md](./README.md) for pattern structure and [CONTRIBUTING.md](../../../CONTRIBUTING.md) for development workflow.

**Before adding a pattern, verify:**
1. Bug causes **silent wrong results** (not crashes or style issues)
2. Bug is **common** in real scientific code
3. Bug is **detectable** by a constrained LLM (not requiring deep domain knowledge)
4. Pattern doesn't duplicate existing coverage
5. Library is used by scientists/researchers (not just production ML)

---

## References

Research informing these priorities:

- [PyTorch vs TensorFlow Comparative Survey (2025)](https://arxiv.org/html/2508.04035v1) - Framework usage statistics
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) - Original paper, 30k+ citations
- [Gradient Boosting in Scientific Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC9647306/) - Medical AI applications
- [Deep Learning Framework Comparison](https://www.infoworld.com/article/2336447/tensorflow-pytorch-and-jax-choosing-a-deep-learning-framework.html) - Industry analysis
- [Machine Learning Libraries Survey 2025](https://machinelearningmastery.com/10-must-know-python-libraries-for-machine-learning-in-2025/) - Library landscape
