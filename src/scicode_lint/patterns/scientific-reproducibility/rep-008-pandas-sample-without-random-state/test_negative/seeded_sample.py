"""Using numpy Generator for reproducible sampling instead of pandas."""

import numpy as np
import pandas as pd


class ReproducibleSampler:
    """Sampler using numpy Generator for consistent reproducibility."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_rows(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample n rows reproducibly using pre-seeded generator."""
        indices = self.rng.choice(len(df), size=n, replace=False)
        return df.iloc[indices]

    def bootstrap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bootstrap sample using pre-seeded generator."""
        indices = self.rng.choice(len(df), size=len(df), replace=True)
        return df.iloc[indices]


def create_stratified_sample(
    df: pd.DataFrame,
    group_col: str,
    n_per_group: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Create stratified sample using numpy generator per group."""
    rng = np.random.default_rng(seed)
    samples = []
    for _, group in df.groupby(group_col):
        indices = rng.choice(len(group), size=min(n_per_group, len(group)), replace=False)
        samples.append(group.iloc[indices])
    return pd.concat(samples, ignore_index=True)
