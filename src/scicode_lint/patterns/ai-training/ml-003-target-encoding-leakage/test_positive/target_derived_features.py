import pandas as pd


class FeatureBuilder:
    """Feature builder for target-based features."""

    def __init__(self, target_col: str = "target"):
        self.target_col = target_col
        self.category_means: dict = {}
        self.global_stats: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and apply target-based features on full dataset."""
        result = df.copy()

        self.global_stats["mean"] = df[self.target_col].mean()
        self.global_stats["std"] = df[self.target_col].std()

        result["target_zscore"] = (
            df[self.target_col] - self.global_stats["mean"]
        ) / self.global_stats["std"]

        for col in df.select_dtypes(include=["object", "category"]).columns:
            self.category_means[col] = df.groupby(col)[self.target_col].mean().to_dict()
            result[f"{col}_target_mean"] = df[col].map(self.category_means[col])

        return result


def build_interaction_features(df: pd.DataFrame, target: str):
    """Create interaction features from target variable."""
    df["target_percentile"] = df[target].rank(pct=True)

    threshold = df[target].median()
    df["above_median"] = (df[target] > threshold).astype(int)

    df_sorted = df.sort_values(target)
    df_sorted["cumulative_target_mean"] = df_sorted[target].expanding().mean()

    return df_sorted.drop(columns=[target])
