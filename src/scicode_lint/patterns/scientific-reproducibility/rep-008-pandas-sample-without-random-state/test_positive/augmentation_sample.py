"""Data augmentation using pandas sample without random_state."""

import pandas as pd


class DataAugmenter:
    """Augment dataset by sampling and transforming."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def oversample_minority(self, label_col: str, target_count: int) -> pd.DataFrame:
        """Oversample minority class - non-reproducible without random_state."""
        minority = self.df[self.df[label_col] == 1]
        oversampled = minority.sample(n=target_count, replace=True)
        return pd.concat([self.df, oversampled])

    def create_bootstrap_samples(self, n_bootstraps: int = 10) -> list[pd.DataFrame]:
        """Create bootstrap samples - non-reproducible."""
        return [self.df.sample(frac=1.0, replace=True) for _ in range(n_bootstraps)]
