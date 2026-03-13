from sklearn.model_selection import train_test_split


def feature_engineering_without_target(df):
    df["feature_ratio"] = df["feature1"] / (df["feature2"] + 1)
    df["feature_sum"] = df["feature1"] + df["feature2"]

    X = df[["feature1", "feature2", "feature_ratio", "feature_sum"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def category_encoding_without_target(df):
    df["category_freq"] = df.groupby("category")["category"].transform("count")

    X = df[["feature1", "category_freq"]]
    y = df["target"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
