"""Reproducible experiment setup using configuration object."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""

    seed: int = 42
    n_samples: int = 1000
    test_size: float = 0.2

    def setup_rng(self) -> np.random.Generator:
        """Create reproducible random generator."""
        return np.random.default_rng(self.seed)


def run_reproducible_analysis(config: ExperimentConfig):
    """Run analysis with reproducible random state from config."""
    rng = config.setup_rng()

    X = rng.standard_normal((config.n_samples, 10))
    y = rng.integers(0, 2, config.n_samples)

    return X, y


if __name__ == "__main__":
    config = ExperimentConfig(seed=123)
    X, y = run_reproducible_analysis(config)
    print(f"Generated {len(X)} samples reproducibly")
