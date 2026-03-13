from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2, seed=42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def stratified_split(df, target_col, test_size=0.2, seed=42):
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=seed
    )
    return train, test


class DataSplitter:
    def __init__(self, seed=42):
        self.seed = seed

    def split(self, df, test_size=0.2):
        return train_test_split(df, test_size=test_size, random_state=self.seed)

    def kfold_split(self, df, n_splits=5):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        return list(kf.split(df))
