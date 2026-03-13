def create_features(df):
    df["next_value"] = df["target"].shift(-1)
    df["target_rolling"] = df["target"].rolling(3).mean()
    return df.dropna()


def add_target_stats(X, y):
    X["target_mean"] = y.mean()
    X["target_std"] = y.std()
    return X
