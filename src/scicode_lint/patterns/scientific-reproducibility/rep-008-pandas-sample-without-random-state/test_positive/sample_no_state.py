"""Feature engineering pipeline with non-reproducible sampling."""

import pandas as pd


class FeatureEngineer:
    """Build features using sampled reference distributions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def subsample_data(self, n: int = 500) -> pd.DataFrame:
        """Subsample for fast prototyping - non-reproducible."""
        return self.df.sample(n=n)

    def compute_reference_stats(self, frac: float = 0.3) -> dict:
        """Compute stats from random subset - non-reproducible."""
        ref = self.df.sample(frac=frac)
        return {"mean": ref["value"].mean(), "std": ref["value"].std()}

    def stratified_downsample(self, group_col: str, n_per_group: int) -> pd.DataFrame:
        """Downsample each group - non-reproducible."""
        return self.df.groupby(group_col).apply(
            lambda g: g.sample(n=min(len(g), n_per_group))
        )
