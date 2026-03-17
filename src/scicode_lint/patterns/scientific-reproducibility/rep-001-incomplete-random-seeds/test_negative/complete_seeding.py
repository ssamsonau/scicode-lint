"""Complete reproducibility setup using seed context manager."""

import contextlib
import os
import random

import numpy as np
import torch


@contextlib.contextmanager
def reproducible_context(seed: int = 42):
    """Context manager that ensures complete reproducibility."""
    old_random_state = random.getstate()
    old_np_state = np.random.get_state()
    old_torch_state = torch.get_rng_state()
    old_env = os.environ.get("PYTHONHASHSEED")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        old_cuda_state = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        yield
    finally:
        random.setstate(old_random_state)
        np.random.set_state(old_np_state)
        torch.set_rng_state(old_torch_state)
        if old_env is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_env

        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(old_cuda_state)


def run_reproducible_experiment(seed: int = 42):
    """Run experiment with full reproducibility guarantees."""
    with reproducible_context(seed):
        data = np.random.randn(100, 10)
        model = torch.nn.Linear(10, 1)
        return data, model
