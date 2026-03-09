def filter_data(df, condition):
    return df[condition]


def head_n_rows(df, n=1000):
    return df.head(n)


def slice_data(df, start, end):
    return df.iloc[start:end]


def select_by_index(df, indices):
    return df.iloc[indices]


class DataSelector:
    def __init__(self, df):
        self.df = df

    def get_by_label(self, label_col, labels):
        return self.df[self.df[label_col].isin(labels)]

    def get_first_n(self, n):
        return self.df.head(n)

    def get_every_nth(self, n):
        return self.df.iloc[::n]
