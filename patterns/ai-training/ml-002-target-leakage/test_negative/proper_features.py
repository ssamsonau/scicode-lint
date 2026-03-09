def create_features(df):
    df["prev_value"] = df["feature1"].shift(1)
    df["feature_rolling"] = df["feature1"].rolling(3).mean()
    return df.dropna()


def add_feature_stats(X):
    X["feature_mean"] = X["feature1"].mean()
    X["feature_std"] = X["feature1"].std()
    return X
