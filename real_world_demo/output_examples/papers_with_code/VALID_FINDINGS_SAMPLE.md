# Valid Findings - Quick Verification Sample

**10 verified findings** (4 critical + 6 high) with pattern diversity for fast manual verification.

## 1. ml-009 (critical)

**File:** [datasets/aws.py](https://github.com/sheoyon-jhin/contime/blob/main/datasets/aws.py)
**Repo:** [sheoyon-jhin__contime](https://github.com/sheoyon-jhin/contime)
**Paper:** Addressing Prediction Delays in Time Series Forecasting: A Continuous GRU Approach with Derivative Regularization
**Authors:** Sheo Yon Jhin et al.

**Issue:** ml-009: Issue detected

**Explanation:** Overlap leakage: Train and test data derive from the same source without guaranteed disjoint splitting. Use sklearn.model_selection.train_test_split() or verify indices are disjoint.


**Verification reasoning:** VALID: This is a real issue that should be fixed

The code creates overlapping sequences from a single time series using `get_sequences()` with a sliding window approach, then splits these sequences into train/val/test sets using `split_data()`. With sliding windows (especially with `stride_window` potentially smaller than `look_window + forecast_window`), adjacent sequences share data points, meaning train and test sequences likely contain overlapping time periods. This is a classic temporal leakage problem - the model can memorize patterns from shared data points that appear in both training and test sets, leading to overly optimistic evaluation metrics.

---

## 2. num-001 (critical)

**File:** [deepcde/tests.py](https://github.com/tpospisi/DeepCDE/blob/main/deepcde/tests.py)
**Repo:** [tpospisi__DeepCDE](https://github.com/tpospisi/DeepCDE)
**Paper:** Conditional Density Estimation Tools in Python and R with Applications to Photometric Redshifts and Likelihood-Free Cosmological Inference
**Authors:** Niccolò Dalmasso et al.

**Issue:** num-001: Issue detected

**Explanation:** Never use == on floats. 0.1 + 0.2 != 0.3 in floating point. Use np.isclose() or np.allclose() with appropriate tolerance.



**Verification reasoning:** VALID: This is a real issue that should be fixed

The code uses direct equality comparisons on floating-point values in three places: `(box_transform(...).round(2) == example_array).all()` comparisons and `np.sum(example_density[idx_negative]) == 0`. While `.round(2)` mitigates some floating-point precision issues, it doesn't eliminate them entirely, and comparing sums to exactly zero is problematic. These should use `np.allclose()` or `np.isclose()` with appropriate tolerances for more robust numerical comparisons in scientific code.

---

## 3. ml-010 (critical)

**File:** [XGBOOST_model.ipynb](https://github.com/juliabringewald/Solar-Flare-Forecast/blob/main/XGBOOST_model.ipynb)
**Repo:** [juliabringewald__Solar-Flare-Forecast](https://github.com/juliabringewald/Solar-Flare-Forecast)
**Paper:** Solar Flare Forecast: A Comparative Analysis of Machine Learning Algorithms for Solar Flare Class Prediction
**Authors:** Julia Bringewald

**Issue:** ml-010: Issue detected

**Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.


**Verification reasoning:** VALID: This is a real issue that should be fixed

The code uses `StratifiedKFold` cross-validation with `GridSearchCV` for hyperparameter tuning, but there is no held-out test set visible in the code context. The same cross-validation splits appear to be used for both model selection/tuning and final evaluation, which constitutes multi-test leakage where the "test" data influences model development. A proper held-out test set that is never seen during any part of model development should be created before cross-validation begins.

---

## 4. pt-003 (critical)

**File:** [utils/transform.py](https://github.com/jjzhou012/I2BGNN/blob/main/utils/transform.py)
**Repo:** [jjzhou012__I2BGNN](https://github.com/jjzhou012/I2BGNN)
**Paper:** Identity Inference on Blockchain using Graph Neural Network
**Authors:** Jie Shen et al.

**Issue:** pt-003: Issue detected

**Explanation:** In-place operation on leaf variable: In-place operations modify tensors with requires_grad=True, breaking the autograd graph. This will cause a runtime error during backward(). Use non-in-place operations instead (e.g., x = x + y instead of x.add_(y)).


**Verification reasoning:** VALID: This is a real issue that should be fixed

The code uses in-place operations `div_()` and `clamp_()` on `value` which is a tensor attribute from PyTorch Geometric data. If this tensor has `requires_grad=True` (which is common for feature tensors in training), these in-place operations will corrupt the autograd graph and cause runtime errors during backpropagation. The fix would be to use `value = value / value.sum(dim=0, keepdim=True).clamp(min=1.)` instead.

---

## 5. rep-002 (high)

**File:** [stage3.py](https://github.com/lsh0520/3d-molm/blob/main/stage3.py)
**Repo:** [lsh0520__3d-molm](https://github.com/lsh0520/3d-molm)
**Paper:** Towards 3D Molecule-Text Interpretation in Language Models
**Authors:** Sihang Li et al.

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Verification reasoning:** VALID: This is a real issue that should be fixed

The code uses `pl.seed_everything(args.seed)` for reproducibility but does not set `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.benchmark = False`. This means GPU operations like convolutions and certain tensor operations may produce non-deterministic results across runs even with the same seed, undermining reproducibility in this deep learning training script.

---

## 6. pt-009 (high)

**File:** [src/ace_inference/core/aggregator/one_step/reduced.py](https://github.com/rose-stl-lab/spherical-dyffusion/blob/main/src/ace_inference/core/aggregator/one_step/reduced.py)
**Repo:** [rose-stl-lab__spherical-dyffusion](https://github.com/rose-stl-lab/spherical-dyffusion)
**Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
**Authors:** Salva Rühling Cachay et al.

**Issue:** pt-009: Issue detected

**Explanation:** Loss tensor accumulation: Accumulating loss tensors directly keeps entire computation graphs in memory, causing memory leaks. Extract the scalar value using loss.item() or loss.detach().cpu().item() before accumulating.


**Verification reasoning:** VALID: This is a real issue that should be fixed

The `L1Loss.record()` method accumulates loss tensors directly with `self._total += nn.functional.l1_loss(...)`, which keeps the entire computation graph attached to `_total`. Over multiple calls, this causes memory to grow unboundedly. The fix is to use `.item()` or `.detach()` before accumulation: `self._total += nn.functional.l1_loss(gen, target).detach()`.

---

## 7. num-003 (high)

**File:** [model/pygon_runner.py](https://github.com/louzounlab/pygon/blob/main/model/pygon_runner.py)
**Repo:** [louzounlab__pygon](https://github.com/louzounlab/pygon)
**Paper:** Planted Dense Subgraphs in Dense Random Graphs Can Be Recovered using Graph-based Machine Learning
**Authors:** Itay Levinas, yoram louzoun

**Issue:** num-003: Issue detected

**Explanation:** np.log(0) returns -inf silently and poisons downstream computations. Use np.log1p(x) for log(1+x), or add a guard/clamp.



**Verification reasoning:** VALID: This is a real issue that should be fixed

The `_binomial_reg` method uses `np.log()` on expressions like `self._graph_params["subgraph_size"] / self._graph_params["vertices"]` without any guards. If `subgraph_size` is 0 or equals `vertices`, the log arguments become 0, producing `-inf` that will poison the loss computation. Unlike `_pairwise_loss` which properly guards against this with `torch.where` and a minimum value of `1e-8`, `_binomial_reg` has no such protection.

---

## 8. rep-001 (high)

**File:** [mksubset.py](https://github.com/r9y9/wavenet_vocoder/blob/main/mksubset.py)
**Repo:** [r9y9__wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
**Paper:** Parametric Resynthesis with neural vocoders
**Authors:** Soumi Maiti, Michael I Mandel

**Issue:** rep-001: Issue detected

**Explanation:** Incomplete random seeding detected. Seed all random libraries: np.random.seed(), random.seed(), torch.manual_seed(), and sklearn random_state parameters.


**Verification reasoning:** VALID: This is a real issue that should be fixed

The script accepts a `--random-state` parameter (defaulting to 1234) but the code shown never calls `np.random.seed()` or `random.seed()` with this value. When `train_dev_test_split` is used, the data splitting will likely involve random operations (presumably using sklearn's train_test_split or similar), and without proper seeding, the splits won't be reproducible even though the user explicitly provided a random state parameter.

---

## 9. perf-001 (high)

**File:** [module_train/main_train.py](https://github.com/ArnaudFerre/C-Norm/blob/main/module_train/main_train.py)
**Repo:** [ArnaudFerre__C-Norm](https://github.com/ArnaudFerre/C-Norm)
**Paper:** C-Norm: a neural approach to few-shot entity normalization
**Authors:** Arnaud Ferré et al.

**Issue:** perf-001: Issue detected

**Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.



**Verification reasoning:** VALID: This is a real issue that should be fixed

The code iterates over a dictionary containing NumPy arrays and performs element-wise division in a Python loop (`for token in vst_onlyTokens.keys()`). This could be vectorized by stacking all embeddings into a 2D array and normalizing in a single NumPy operation using broadcasting, which would be significantly faster for large vocabularies. While the loop itself is over dictionary keys rather than array elements, the normalization operation inside could benefit from batch vectorization if the embeddings were stored as a matrix instead of a dictionary.

---

## 10. pt-015 (high)

**File:** [evautils_PCAE_on_SN.py](https://github.com/GentleDell/Better-Patch-Stitching/blob/main/evautils_PCAE_on_SN.py)
**Repo:** [GentleDell__Better-Patch-Stitching](https://github.com/GentleDell/Better-Patch-Stitching)
**Paper:** Better Patch Stitching for Parametric Surface Reconstruction
**Authors:** Zhantao Deng et al.

**Issue:** pt-015: Issue detected

**Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.


**Verification reasoning:** VALID: This is a real issue that should be fixed

The code uses `torch.load(path_weight)` without specifying `map_location`, which will fail on CPU-only systems if the model was saved on a GPU. The fix is straightforward: change to `torch.load(path_weight, map_location='cpu')` or use the device variable that's determined on the next line.

---
