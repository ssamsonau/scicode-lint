# Real-World Scientific ML Code Analysis Report (Verified Findings Only)

> **⚠️ Results Pending Re-evaluation**
>
> This analysis was run with a bug that reduced detection accuracy from ~99% to ~78%.
> The bug caused `response_format: json_schema` to skip Qwen3's thinking phase.
> Results are expected to improve after re-running with the fix (commit 39847fa).
> See `src/scicode_lint/llm/client.py` module docstring for details.

Analysis of Python code from **AI applications to scientific domains**. Papers sourced from PapersWithCode, filtered to include only scientific domains (biology, chemistry, physics, medicine, earth science, astronomy, materials, etc.) where ML/AI is applied to scientific discovery and domain-specific research.

## Analysis Summary

- **Analysis Date:** 2026-03-12 18:52
- **Report Generated:** 2026-03-12 23:36
- **Papers with Findings:** 38 / 38 (100.0%)
- **Repos with Findings:** 48 / 48 (100.0%)
- **Files Analyzed:** 884 / 884
- **Files with Findings:** 530 (60.0%)
- **Total Findings:** 1,709

## Papers by Severity

Papers with at least one finding of each severity level (a paper may appear in multiple rows):

| Severity | Papers | % of Papers Analyzed |
|----------|--------|----------------------|
| Critical | 35 | 92.1% |
| High | 37 | 97.4% |
| Medium | 35 | 92.1% |

*Total papers analyzed: 38*

## Findings Distribution (per paper)

| Metric | All | Critical | High | Medium | Low |
|--------|-----|----------|------|--------|-----|
| Papers | 38 | 35 | 37 | 35 | 0 |
| Min | 2 | 1 | 1 | 1 | - |
| Max | 346 | 42 | 186 | 118 | - |
| Mean | 45.0 | 7.2 | 24.8 | 15.5 | - |
| Median | 28.0 | 5.0 | 15.0 | 11.0 | - |
| Std Dev | 64.0 | 8.6 | 36.4 | 20.9 | - |

## Verification Summary

**Overall Precision:** 18.1% (309 valid / 1,709 verified)

| Status | Count | % |
|--------|-------|---|
| Valid (confirmed) | 309 | 18.1% |
| Invalid (false positive) | 1,303 | 76.2% |
| Uncertain | 97 | 5.7% |

### Verified Findings by Severity

| Severity | Total | Valid | Invalid | Uncertain | Pending | Precision |
|----------|-------|-------|---------|-----------|---------|-----------|
| Critical | 251 | 8 | 180 | 63 | 0 | 3% |
| High | 916 | 157 | 733 | 26 | 0 | 17% |
| Medium | 542 | 144 | 390 | 8 | 0 | 27% |

## Findings by Scientific Domain

| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |
|--------|---------------|---------------|--------------|----------------|
| chemistry | 216 | 169 | 78.2% | 633 |
| earth_science | 257 | 95 | 37.0% | 257 |
| none | 50 | 43 | 86.0% | 142 |
| materials | 53 | 37 | 69.8% | 132 |
| economics | 94 | 48 | 51.1% | 114 |
| biology | 54 | 45 | 83.3% | 109 |
| medicine | 46 | 25 | 54.3% | 97 |
| engineering | 38 | 23 | 60.5% | 88 |
| astronomy | 15 | 14 | 93.3% | 55 |
| physics | 21 | 16 | 76.2% | 49 |
| neuroscience | 15 | 9 | 60.0% | 20 |
| social_science | 6 | 4 | 66.7% | 11 |
| mathematics | 6 | 2 | 33.3% | 2 |

## Findings by Category

| Category | Count | Unique Files | Unique Repos |
|----------|-------|--------------|--------------|
| scientific-reproducibility | 712 | 389 | 45 |
| scientific-performance | 379 | 227 | 41 |
| ai-training | 373 | 261 | 43 |
| ai-inference | 154 | 121 | 30 |
| scientific-numerical | 91 | 75 | 28 |

## Findings by Severity

| Severity | Count | % of Total |
|----------|-------|------------|
| Critical | 251 | 14.7% |
| High | 916 | 53.6% |
| Medium | 542 | 31.7% |

## Most Common Patterns

| Pattern | Category | Count | Files | Repos | Avg Confidence |
|---------|----------|-------|-------|-------|----------------|
| rep-002 | scientific-reproducibility | 238 | 238 | 34 | 95% |
| ml-010 | ai-training | 215 | 215 | 42 | 95% |
| rep-004 | scientific-reproducibility | 146 | 146 | 34 | 95% |
| perf-001 | scientific-performance | 130 | 130 | 36 | 95% |
| rep-003 | scientific-reproducibility | 126 | 126 | 35 | 95% |
| pt-010 | ai-training | 116 | 116 | 29 | 95% |
| perf-002 | scientific-performance | 92 | 92 | 29 | 95% |
| rep-001 | scientific-reproducibility | 73 | 73 | 23 | 95% |
| rep-010 | scientific-reproducibility | 60 | 60 | 17 | 95% |
| pt-015 | ai-inference | 58 | 58 | 18 | 95% |

## Example Findings

Representative findings from each category (with links to source):

### ai-inference

**pt-013** (medium, 95% confidence)

- **File:** [src/ms_pred/gnn_pred/predict.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/gnn_pred/predict.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** pt-013: Issue detected
- **Explanation:** torch.inference_mode() is faster than no_grad() for production inference. Use inference_mode() when gradients will never be needed.

- **Suggestion:** Review the code and fix according to the explanation.

**pt-014** (medium, 95% confidence)

- **File:** [src/ms_pred/gnn_pred/predict.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/gnn_pred/predict.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** pt-014: Issue detected
- **Explanation:** CUDA timing without torch.cuda.synchronize() is inaccurate. Call synchronize() before measuring end time to ensure all GPU operations complete.

- **Suggestion:** Review the code and fix according to the explanation.

**pt-015** (high, 95% confidence)

- **File:** [visualization.py](https://github.com/shmily-ld/dynamicdta/blob/main/visualization.py) in [shmily-ld__dynamicdta](https://github.com/shmily-ld/dynamicdta)
- **Paper:** DynamicDTA: Drug-Target Binding Affinity Prediction Using Dynamic Descriptors and Graph Representation
- **Authors:** Dan Luo et al.
- **Issue:** pt-015: Issue detected
- **Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.

- **Suggestion:** Review the code and fix according to the explanation.

### ai-training

**ml-010** (critical, 95% confidence)

- **File:** [src/ms_pred/autoregr_gen/hyperopt.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/autoregr_gen/hyperopt.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** ml-010: Issue detected
- **Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.

- **Suggestion:** Review the code and fix according to the explanation.

**ml-010** (critical, 95% confidence)

- **File:** [tool/dataset.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/main/tool/dataset.py) in [MLRG-CEFET-RJ__stconvs2s](https://github.com/MLRG-CEFET-RJ/stconvs2s)
- **Paper:** STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting
- **Authors:** Rafaela Castro et al.
- **Issue:** ml-010: Issue detected
- **Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.

- **Suggestion:** Review the code and fix according to the explanation.

**pt-010** (high, 95% confidence)

- **File:** [src/ms_pred/gnn_pred/train.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/gnn_pred/train.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** pt-010: Issue detected
- **Explanation:** DataLoader with num_workers=0 forces single-process data loading, creating a bottleneck where the GPU idles while waiting for data. Set num_workers to 2-8 (based on CPU cores) to enable multi-process data loading with prefetching, significantly improving training throughput.


- **Suggestion:** Review the code and fix according to the explanation.

### scientific-numerical

**num-006** (medium, 95% confidence)

- **File:** [cgnet/tests/test_geometry_features.py](https://github.com/coarse-graining/cgnet/blob/main/cgnet/tests/test_geometry_features.py) in [coarse-graining__cgnet](https://github.com/coarse-graining/cgnet)
- **Paper:** Coarse Graining Molecular Dynamics with Graph Neural Networks
- **Authors:** Brooke E. Husic et al.
- **Issue:** num-006: Issue detected
- **Explanation:** Use np.testing.assert_allclose(actual, expected, rtol, atol) for float comparisons in tests. Exact equality assertions on floats are unreliable.


- **Suggestion:** Review the code and fix according to the explanation.

**num-006** (medium, 95% confidence)

- **File:** [cgnet/tests/test_schnet_features.py](https://github.com/coarse-graining/cgnet/blob/main/cgnet/tests/test_schnet_features.py) in [coarse-graining__cgnet](https://github.com/coarse-graining/cgnet)
- **Paper:** Coarse Graining Molecular Dynamics with Graph Neural Networks
- **Authors:** Brooke E. Husic et al.
- **Issue:** num-006: Issue detected
- **Explanation:** Use np.testing.assert_allclose(actual, expected, rtol, atol) for float comparisons in tests. Exact equality assertions on floats are unreliable.


- **Suggestion:** Review the code and fix according to the explanation.

**num-003** (high, 95% confidence)

- **File:** [notebooks/figureS5.ipynb](https://github.com/citrineinformatics-erd-public/piml_glass_forming_ability/blob/main/notebooks/figureS5.ipynb) in [citrineinformatics-erd-public__piml_glass_forming_ability](https://github.com/citrineinformatics-erd-public/piml_glass_forming_ability)
- **Paper:** Evaluation of GlassNet for physics-informed machine learning of glass stability and glass-forming ability
- **Authors:** Sarah I. Allec et al.
- **Issue:** num-003: Issue detected
- **Explanation:** np.log(0) returns -inf silently and poisons downstream computations. Use np.log1p(x) for log(1+x), or add a guard/clamp.


- **Suggestion:** Review the code and fix according to the explanation.

### scientific-performance

**perf-001** (high, 95% confidence)

- **File:** [cgnet/tests/test_geometry_features.py](https://github.com/coarse-graining/cgnet/blob/main/cgnet/tests/test_geometry_features.py) in [coarse-graining__cgnet](https://github.com/coarse-graining/cgnet)
- **Paper:** Coarse Graining Molecular Dynamics with Graph Neural Networks
- **Authors:** Brooke E. Husic et al.
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

**perf-002** (high, 95% confidence)

- **File:** [inference.ipynb](https://github.com/lsh0520/3d-molm/blob/main/inference.ipynb) in [lsh0520__3d-molm](https://github.com/lsh0520/3d-molm)
- **Paper:** Towards 3D Molecule-Text Interpretation in Language Models
- **Authors:** Sihang Li et al.
- **Issue:** perf-002: Issue detected
- **Explanation:** Array allocation in loop: creating arrays inside loops causes repeated malloc/free operations. Pre-allocate the output array before the loop and fill it using indexing.

- **Suggestion:** Review the code and fix according to the explanation.

**perf-001** (high, 95% confidence)

- **File:** [src/ms_pred/gnn_pred/predict.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/gnn_pred/predict.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

### scientific-reproducibility

**rep-004** (medium, 100% confidence)

- **File:** [FusionRetro/reward_model.py](https://github.com/songtaoliu0823/crebm/blob/main/FusionRetro/reward_model.py) in [songtaoliu0823__crebm](https://github.com/songtaoliu0823/crebm)
- **Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
- **Authors:** Songtao Liu et al.
- **Issue:** rep-004: Issue detected
- **Explanation:** No random seed set. Results will differ between runs. Set seeds for np.random, torch, and random at the start.

- **Suggestion:** Review the code and fix according to the explanation.

**rep-002** (high, 95% confidence)

- **File:** [src/ms_pred/autoregr_gen/hyperopt.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/autoregr_gen/hyperopt.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** rep-002: Issue detected
- **Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.


- **Suggestion:** Review the code and fix according to the explanation.

**rep-003** (medium, 95% confidence)

- **File:** [src/ms_pred/autoregr_gen/hyperopt.py](https://github.com/samgoldman97/ms-pred/blob/main/src/ms_pred/autoregr_gen/hyperopt.py) in [samgoldman97__ms-pred](https://github.com/samgoldman97/ms-pred)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** rep-003: Issue detected
- **Explanation:** Hardcoded magic numbers: parameters buried in code can't be tracked or reproduced. Use a config file or argparse for all hyperparameters.


- **Suggestion:** Review the code and fix according to the explanation.

---

*Analysis conducted: 2026-03-12 | Report generated: 2026-03-12 23:36*
