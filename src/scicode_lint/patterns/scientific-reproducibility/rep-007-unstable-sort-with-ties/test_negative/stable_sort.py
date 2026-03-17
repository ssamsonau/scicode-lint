import numpy as np
import pandas as pd


def rank_with_pandas(df: pd.DataFrame, column: str) -> pd.Series:
    sorted_df = df.sort_values(column, kind="mergesort")
    return sorted_df.index


def get_top_k_with_tiebreaker(
    values: np.ndarray,
    secondary_key: np.ndarray,
    k: int,
) -> np.ndarray:
    df = pd.DataFrame({"value": values, "tiebreaker": secondary_key})
    sorted_df = df.sort_values(
        ["value", "tiebreaker"],
        ascending=[False, True],
    )
    return sorted_df.head(k).index.values


class TieBreakerRanker:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.indices = np.arange(len(data))

    def rank(self) -> np.ndarray:
        order = np.lexsort((self.indices, self.data))
        return order
