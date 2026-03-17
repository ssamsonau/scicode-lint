import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    learning_rates: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    tags: set = field(default_factory=set)


def create_pipeline(steps=None):
    if steps is None:
        steps = [("scaler", "standard"), ("model", "linear")]
    return steps


def bootstrap_sample(data, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    indices = rng.integers(0, len(data), size=len(data))
    return data[indices]


def cross_validate(model, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(100)
    fold_size = len(indices) // folds
    return [indices[i * fold_size:(i + 1) * fold_size] for i in range(folds)]


def compute_stats(values, *, axis=0, ddof=1):
    arr = np.asarray(values)
    return {"mean": arr.mean(axis=axis), "std": arr.std(axis=axis, ddof=ddof)}
