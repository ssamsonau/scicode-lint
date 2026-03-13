# Real-World Scientific ML Code Analysis Report

Analysis of Python code from scientific ML papers using scicode-lint. Papers sourced from PapersWithCode, filtered by scientific domain.


> **Verification (Claude Opus, 2026-03-11 22:57):** 1/21 findings confirmed as real issues (5% precision)

## Analysis Summary

- **Analysis Date:** 2026-03-11 21:26
- **Report Generated:** 2026-03-11 22:56
- **Repositories Analyzed:** 56
- **Files Analyzed:** 876 / 1,498
- **Files with Findings:** 16 (1.8%)
- **Total Findings:** 21

## Findings by Scientific Domain

| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |
|--------|---------------|---------------|--------------|----------------|
| physics | 14 | 5 | 35.7% | 9 |
| biology | 44 | 6 | 13.6% | 7 |
| earth_science | 6 | 2 | 33.3% | 2 |
| medical | 5 | 1 | 20.0% | 1 |
| chemistry | 2 | 1 | 50.0% | 1 |
| materials | 1 | 1 | 100.0% | 1 |
| neuroscience | 2 | 0 | 0.0% | 0 |
| astronomy | 1 | 0 | 0.0% | 0 |

## Findings by Category

| Category | Count | Unique Files | Unique Repos |
|----------|-------|--------------|--------------|
| scientific-reproducibility | 9 | 9 | 6 |
| ai-training | 5 | 5 | 5 |
| scientific-performance | 3 | 3 | 2 |
| ai-inference | 3 | 3 | 1 |
| scientific-numerical | 1 | 1 | 1 |

## Findings by Severity

- **Critical:** 1
- **High:** 11
- **Medium:** 9

## Most Common Patterns

| Pattern | Category | Count | Files | Repos | Avg Confidence |
|---------|----------|-------|-------|-------|----------------|
| rep-003 | scientific-reproducibility | 6 | 6 | 4 | 95% |
| pt-006 | ai-inference | 3 | 3 | 1 | 78% |
| perf-001 | scientific-performance | 3 | 3 | 2 | 95% |
| pt-008 | ai-training | 2 | 2 | 2 | 70% |
| rep-011 | scientific-reproducibility | 1 | 1 | 1 | 95% |
| rep-010 | scientific-reproducibility | 1 | 1 | 1 | 95% |
| rep-002 | scientific-reproducibility | 1 | 1 | 1 | 95% |
| py-001 | scientific-numerical | 1 | 1 | 1 | 95% |
| pt-012 | ai-training | 1 | 1 | 1 | 70% |
| pt-004 | ai-training | 1 | 1 | 1 | 70% |

## Example Findings

Representative findings from each category (with links to source):

### ai-inference

**pt-006** [FALSE POSITIVE]  (high, 95% confidence)

- **File:** [rllab/sandbox/rocky/tf/q_functions/continuous_mlp_q_function.py](https://github.com/mengyuest/reactive_cbf/blob/main/rllab/sandbox/rocky/tf/q_functions/continuous_mlp_q_function.py#L60-L76) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** pt-006: Issue detected
- **Explanation:** Thresholding raw logits is meaningless — logits are unbounded. Apply torch.sigmoid() first, then threshold.


- **Suggestion:** Review the code and fix according to the explanation.

```python
nonlinearity=output_nonlinearity,
        output_var = L.get_output(l_output, deterministic=True)
```

**pt-006** [FALSE POSITIVE]  (high, 70% confidence)

- **File:** [hgail/scripts/utils.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/scripts/utils.py#L1-L3) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** pt-006: Issue detected
- **Explanation:** Thresholding raw logits is meaningless — logits are unbounded. Apply torch.sigmoid() first, then threshold.


- **Suggestion:** Review the code and fix according to the explanation.

```python
import datetime
import glob
```

**pt-006** [FALSE POSITIVE]  (high, 70% confidence)

- **File:** [hgail/hgail/policies/categorical_latent_sampler.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/hgail/policies/categorical_latent_sampler.py#L36-L60) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** pt-006: Issue detected
- **Explanation:** Thresholding raw logits is meaningless — logits are unbounded. Apply torch.sigmoid() first, then threshold.


- **Suggestion:** Review the code and fix according to the explanation.

```python
prob = self._f_prob([flat_obs])[0]
        actions = list(map(self.action_space.weighted_sample, probs))
```

### ai-training

**pt-008** [FALSE POSITIVE]  (high, 70% confidence)

- **File:** [src/visualize_field.py](https://github.com/hjwdzh/FrameNet/blob/main/src/visualize_field.py#L22-L23) in [hjwdzh__FrameNet](https://github.com/hjwdzh/FrameNet)
- **Paper:** FrameNet: Learning Local Canonical Frames of 3D Surfaces from a Single RGB Image
- **Issue:** pt-008: Issue detected
- **Explanation:** Optimizer step before backward: Calling optimizer.step() before loss.backward() updates parameters with old/zero gradients instead of current gradients. The correct sequence is: optimizer.zero_grad(), compute loss, loss.backward(), then optimizer.step().

- **Suggestion:** Review the code and fix according to the explanation.

```python
train_dataset = AffineTestsDataset(feat=0,root='data')
```

**pt-008** [FALSE POSITIVE]  (high, 70% confidence)

- **File:** [hgail/hgail/misc/flip_gradient.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/hgail/misc/flip_gradient.py#L4-L6) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** pt-008: Issue detected
- **Explanation:** Optimizer step before backward: Calling optimizer.step() before loss.backward() updates parameters with old/zero gradients instead of current gradients. The correct sequence is: optimizer.zero_grad(), compute loss, loss.backward(), then optimizer.step().

- **Suggestion:** Review the code and fix according to the explanation.

```python
class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0
```

**pt-012** [FALSE POSITIVE]  (high, 70% confidence)

- **File:** [metadl/core/run.py](https://github.com/ebadrian/metadl/blob/main/metadl/core/run.py#L66-L70) in [ebadrian__metadl](https://github.com/ebadrian/metadl)
- **Paper:** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
- **Issue:** pt-012: Issue detected
- **Explanation:** Gradient accumulation without loss scaling: accumulating over N steps without dividing loss by N results in gradients N times larger than intended. Divide loss by accumulation_steps before calling backward().

- **Suggestion:** Review the code and fix according to the explanation.

```python
def main(argv):
    """ Runs the ingestion and scoring programs sequentially, as they are 
    del argv
```

### scientific-numerical

**py-001** [FALSE POSITIVE]  (medium, 95% confidence)

- **File:** [meta_dataset/meta_dataset/dataset_conversion/convert_datasets_to_records.py](https://github.com/tmlr-group/CoPA/blob/main/meta_dataset/meta_dataset/dataset_conversion/convert_datasets_to_records.py#L64) in [tmlr-group__CoPA](https://github.com/tmlr-group/CoPA)
- **Paper:** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
- **Issue:** py-001: Issue detected
- **Explanation:** Mutable default argument: the default object is shared across all calls. Use None as default and create inside the function.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def _dataset_name_to_converter_and_args(flags=FLAGS):
```

### scientific-performance

**perf-001** [FALSE POSITIVE]  (high, 95% confidence)

- **File:** [hgail/hgail/policies/categorical_latent_sampler.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/hgail/policies/categorical_latent_sampler.py#L58-L67) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

```python
update_indicators = self._scheduler.should_update(flat_obs)
        update = [False] * len(self._latent_values)
        for (i, indicator) in enumerate(update_indicators):
            if indicator:
                self._latent_values[i] = actions[i]
```

**perf-001** [FALSE POSITIVE]  (high, 95% confidence)

- **File:** [meta_dataset/meta_dataset/data/test_utils.py](https://github.com/tmlr-group/CoPA/blob/main/meta_dataset/meta_dataset/data/test_utils.py#L101) in [tmlr-group__CoPA](https://github.com/tmlr-group/CoPA)
- **Paper:** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

```python
for feat in list(features):
```

**perf-001** [FALSE POSITIVE]  (high, 95% confidence)

- **File:** [meta_dataset/meta_dataset/data/utils.py](https://github.com/tmlr-group/CoPA/blob/main/meta_dataset/meta_dataset/data/utils.py#L42-L43) in [tmlr-group__CoPA](https://github.com/tmlr-group/CoPA)
- **Paper:** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

```python
for image, label in zip(images, labels):
      dataset_to_records.write_example(image.numpy(), label.numpy(), writer)
```

### scientific-reproducibility

**rep-003** [FALSE POSITIVE]  (medium, 95% confidence)

- **File:** [meta_dataset/data/read_episodes.py](https://github.com/tmlr-group/CoPA/blob/main/meta_dataset/data/read_episodes.py#L204-L206) in [tmlr-group__CoPA](https://github.com/tmlr-group/CoPA)
- **Paper:** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
- **Issue:** rep-003: Issue detected
- **Explanation:** Hardcoded magic numbers: parameters buried in code can't be tracked or reproduced. Use a config file or argparse for all hyperparameters.


- **Suggestion:** Review the code and fix according to the explanation.

```python
image_size=224,
                         query_size_limit=500,
                         data_dir=None):
```

**rep-010** [FALSE POSITIVE]  (medium, 95% confidence)

- **File:** [hgail/scripts/utils.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/scripts/utils.py#L16) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** rep-010: Issue detected
- **Explanation:** Naive datetime depends on system timezone. Use datetime.now(timezone.utc) for reproducible timestamps.

- **Suggestion:** Review the code and fix according to the explanation.

```python
x = str(datetime.datetime.now().isoformat())
```

**rep-003** [FALSE POSITIVE]  (medium, 95% confidence)

- **File:** [hgail/hgail/policies/categorical_latent_sampler.py](https://github.com/mengyuest/reactive_cbf/blob/main/hgail/hgail/policies/categorical_latent_sampler.py#L16) in [mengyuest__reactive_cbf](https://github.com/mengyuest/reactive_cbf)
- **Paper:** Reactive and Safe Road User Simulations using Neural Barrier Certificates
- **Issue:** rep-003: Issue detected
- **Explanation:** Hardcoded magic numbers: parameters buried in code can't be tracked or reproduced. Use a config file or argparse for all hyperparameters.


- **Suggestion:** Review the code and fix according to the explanation.

```python
scheduler,
```

---

*Analysis conducted: 2026-03-11 | Verified: 2026-03-11 22:57 (5% precision)*
