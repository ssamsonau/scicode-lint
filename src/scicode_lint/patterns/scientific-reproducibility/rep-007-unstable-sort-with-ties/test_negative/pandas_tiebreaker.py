def sort_with_tiebreaker(df, primary_col, secondary_col):
    return df.sort_values([primary_col, secondary_col])


def rank_predictions(df):
    return df.sort_values(["score", "id"], ascending=[False, True])


def order_by_priority_and_time(df):
    return df.sort_values(["priority", "timestamp"])


class DataFrameSorter:
    def __init__(self, df):
        self.df = df

    def sort_by_columns(self, columns, ascending=True):
        return self.df.sort_values(columns, ascending=ascending)

    def rank_with_index_tiebreaker(self, score_col):
        df = self.df.copy()
        df["_idx"] = range(len(df))
        return df.sort_values([score_col, "_idx"]).drop("_idx", axis=1)
